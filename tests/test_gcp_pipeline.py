"""Cloud actions are mocked; no test connects to GCP or stops a VM."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "gcp_pipeline.py"
spec = importlib.util.spec_from_file_location("gcp_pipeline", MODULE_PATH)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def create_result_archive(path, payload=b"df,delta\n3,2\n"):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("results/test.csv")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))


class DirectCloud:
    target = instance = "test-vm"

    def __init__(self, archive, expected=None, fail_transfer=False):
        self.archive = archive
        self.expected = expected or pipeline.sha256(archive)
        self.fail_transfer = fail_transfer
        self.stopped = False
        self.events = []

    def ssh(self, script, **kwargs):
        return SimpleNamespace(stdout=self.expected + "  results.tar.gz\n", returncode=0)

    def scp(self, source, destination, **kwargs):
        self.events.append("download")
        if self.fail_transfer:
            raise RuntimeError("simulated dropped connection")
        Path(destination).write_bytes(self.archive.read_bytes())

    def stop(self):
        self.events.append("stop")
        self.stopped = True


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_failed_job_download_keeps_vm_running_and_returns_failure(self):
        archive = self.root / "remote.tar.gz"
        create_result_archive(archive)
        cloud = DirectCloud(archive)
        destination = self.root / "local"
        result = pipeline.verified_download_then_stop(cloud, {"remote_dir": "/run"}, destination,
                                                     {"compute_exit_code": 2})
        self.assertEqual(result, 2)
        self.assertEqual(cloud.events, ["download"])
        self.assertFalse(cloud.stopped)
        self.assertTrue((destination / "extracted/results/test.csv").is_file())
        receipt = json.loads((destination / "download.json").read_text())
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["compute_exit_code"], 2)
        self.assertFalse(receipt["stop_requested_after_download"])

    def test_successful_verified_download_precedes_vm_stop(self):
        archive = self.root / "remote.tar.gz"
        create_result_archive(archive)
        cloud = DirectCloud(archive)
        destination = self.root / "local"
        result = pipeline.verified_download_then_stop(cloud, {"remote_dir": "/run"}, destination,
                                                     {"compute_exit_code": 0})
        self.assertEqual(result, 0)
        self.assertEqual(cloud.events, ["download", "stop"])
        self.assertTrue(cloud.stopped)
        receipt = json.loads((destination / "download.json").read_text())
        self.assertTrue(receipt["stop_requested_after_download"])

    def test_prior_backup_error_never_stops_even_when_computation_succeeded(self):
        archive = self.root / "remote.tar.gz"
        create_result_archive(archive)
        cloud = DirectCloud(archive)
        result = pipeline.verified_download_then_stop(cloud, {"remote_dir": "/run"}, self.root / "local",
                                                     {"compute_exit_code": 0, "state": "upload_failed"})
        self.assertEqual(result, 1)
        self.assertFalse(cloud.stopped)

    def test_checksum_mismatch_never_stops_vm(self):
        archive = self.root / "remote.tar.gz"
        create_result_archive(archive)
        cloud = DirectCloud(archive, expected="0" * 64)
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            pipeline.verified_download_then_stop(cloud, {"remote_dir": "/run"}, self.root / "local",
                                                {"compute_exit_code": 0})
        self.assertFalse(cloud.stopped)

    def test_interrupted_download_never_stops_vm(self):
        archive = self.root / "remote.tar.gz"
        create_result_archive(archive)
        cloud = DirectCloud(archive, fail_transfer=True)
        with self.assertRaisesRegex(RuntimeError, "dropped"):
            pipeline.verified_download_then_stop(cloud, {"remote_dir": "/run"}, self.root / "local",
                                                {"compute_exit_code": 0})
        self.assertFalse(cloud.stopped)

    def test_verified_existing_download_is_reused(self):
        archive = self.root / "remote.tar.gz"
        create_result_archive(archive)
        destination = self.root / "local"
        destination.mkdir()
        (destination / "results.tar.gz").write_bytes(archive.read_bytes())
        cloud = DirectCloud(archive)
        pipeline.verified_download_then_stop(cloud, {"remote_dir": "/run"}, destination, {"compute_exit_code": 0})
        self.assertEqual(cloud.events, ["stop"])

    def test_unsafe_tar_is_rejected_before_stop(self):
        archive = self.root / "unsafe.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            member = tarfile.TarInfo("../../outside")
            member.size = 1
            output.addfile(member, io.BytesIO(b"x"))
        cloud = DirectCloud(archive)
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            pipeline.verified_download_then_stop(cloud, {"remote_dir": "/run"}, self.root / "local",
                                                {"compute_exit_code": 0})
        self.assertFalse(cloud.stopped)
        self.assertFalse((self.root.parent / "outside").exists())

    def test_bundle_is_deterministic_and_excludes_environment_cache_credentials(self):
        (self.root / "src").mkdir()
        (self.root / "src/test.py").write_text("print('hello')\n")
        (self.root / "data").mkdir()
        (self.root / "data/input.csv").write_text("df\n3\n")
        (self.root / "data/.env").write_text("SECRET=yes")
        (self.root / "data/key.pem").write_text("PRIVATE")
        (self.root / "data/checkpoints").mkdir()
        (self.root / "data/checkpoints/cache.json").write_text("{}")
        a, b = self.root / "a.tar.gz", self.root / "b.tar.gz"
        fingerprint_a, count, _ = pipeline.create_bundle(self.root, a)
        fingerprint_b, _, _ = pipeline.create_bundle(self.root, b)
        self.assertEqual(count, 2)
        self.assertEqual(fingerprint_a, fingerprint_b)
        self.assertEqual(pipeline.sha256(a), pipeline.sha256(b))
        with tarfile.open(a) as archive:
            self.assertEqual(archive.getnames(), ["data/input.csv", "src/test.py", "source-manifest.json"])

    def test_archived_job_does_not_relaunch_or_upload(self):
        cloud = mock.Mock()
        cloud.ssh.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"state": "archive_ready", "compute_exit_code": 0}))
        pipeline.launch_remote(cloud, {"remote_dir": "/run"}, Path("bundle"), Path("state"))
        cloud.scp.assert_not_called()
        self.assertEqual(cloud.ssh.call_count, 1)

    def test_running_job_does_not_relaunch(self):
        cloud = mock.Mock()
        cloud.ssh.side_effect = [SimpleNamespace(returncode=0, stdout=json.dumps({"state": "running", "pid": 123})),
                                SimpleNamespace(returncode=0, stdout="")]
        pipeline.launch_remote(cloud, {"remote_dir": "/run"}, Path("bundle"), Path("state"))
        cloud.scp.assert_not_called()

    def test_gcs_checksum_failure_does_not_publish_marker_or_stop(self):
        archive = self.root / "results.tar.gz"
        create_result_archive(archive)
        state = {"gcs_prefix": "gs://test/rank", "run_id": "rank-test", "project": "test",
                 "zone": "test-zone", "instance": "test-vm", "shutdown_method": "guest"}
        cloud = mock.Mock()
        process = mock.Mock()
        process.stdout = io.BytesIO(b"corrupt")
        process.wait.return_value = 0
        with mock.patch.object(pipeline, "Cloud", return_value=cloud), \
             mock.patch.object(pipeline.subprocess, "Popen", return_value=process), \
             mock.patch.object(pipeline, "command") as shutdown:
            with self.assertRaisesRegex(RuntimeError, "read-back verification failed"):
                pipeline.upload_and_stop(self.root, state, archive, pipeline.sha256(archive), {"compute_exit_code": 0})
            shutdown.assert_not_called()
        self.assertEqual(cloud.call.call_count, 1)
        self.assertFalse((self.root / "complete.json").exists())

    def test_gcs_success_publishes_marker_before_shutdown(self):
        archive = self.root / "results.tar.gz"
        create_result_archive(archive)
        state = {"gcs_prefix": "gs://test/rank", "run_id": "rank-test", "project": "test",
                 "zone": "test-zone", "instance": "test-vm", "shutdown_method": "guest"}
        events = []
        cloud = mock.Mock()
        cloud.call.side_effect = lambda argv, **kw: events.append("marker" if "complete.json" in str(argv) else "upload")
        process = mock.Mock()
        process.stdout = io.BytesIO(archive.read_bytes())
        process.wait.return_value = 0
        with mock.patch.object(pipeline, "Cloud", return_value=cloud), \
             mock.patch.object(pipeline.subprocess, "Popen", return_value=process), \
             mock.patch.object(pipeline, "command", side_effect=lambda *a, **k: events.append("stop")):
            pipeline.upload_and_stop(self.root, state, archive, pipeline.sha256(archive), {"compute_exit_code": 0})
        self.assertEqual(events, ["upload", "marker", "stop"])
        marker = json.loads((self.root / "complete.json").read_text())
        self.assertEqual(marker["compute_exit_code"], 0)
        self.assertTrue(marker["backup_verified"])
        self.assertTrue(marker["stop_requested_after_backup"])

    def test_gcs_failed_run_backed_up_without_guest_or_gcloud_shutdown(self):
        archive = self.root / "results.tar.gz"
        create_result_archive(archive)
        for shutdown_method in ("guest", "gcloud"):
            with self.subTest(shutdown_method=shutdown_method):
                state = {"gcs_prefix": "gs://test/rank", "run_id": "rank-test", "project": "test",
                         "zone": "test-zone", "instance": "test-vm", "shutdown_method": shutdown_method}
                cloud = mock.Mock()
                process = mock.Mock()
                process.stdout = io.BytesIO(archive.read_bytes())
                process.wait.return_value = 0
                with mock.patch.object(pipeline, "Cloud", return_value=cloud), \
                     mock.patch.object(pipeline.subprocess, "Popen", return_value=process), \
                     mock.patch.object(pipeline, "command") as shutdown:
                    pipeline.upload_and_stop(self.root, state, archive, pipeline.sha256(archive),
                                             {"compute_exit_code": 1})
                    shutdown.assert_not_called()
                self.assertEqual(cloud.call.call_count, 2)
                for call in cloud.call.call_args_list:
                    self.assertEqual(call.args[0][:2], ["storage", "cp"])
                marker = json.loads((self.root / "complete.json").read_text())
                self.assertEqual(marker["compute_exit_code"], 1)
                self.assertTrue(marker["backup_verified"])
                self.assertFalse(marker["stop_requested_after_backup"])

    def test_old_failed_gcs_marker_never_describes_or_stops_vm(self):
        destination = self.root / "download"
        destination.mkdir()
        archive = destination / "results.tar.gz"
        create_result_archive(archive)
        marker = {"sha256": pipeline.sha256(archive), "compute_exit_code": 2,
                  "backup_verified": True, "stop_requested_after_backup": True}
        cloud = mock.Mock()
        cloud.call.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(marker))
        state = {"gcs_prefix": "gs://test/rank", "run_id": "rank-test"}
        self.assertEqual(pipeline.gcs_download(cloud, state, destination), 2)
        cloud.describe.assert_not_called()
        cloud.stop.assert_not_called()
        receipt = json.loads((destination / "download.json").read_text())
        self.assertFalse(receipt["stop_requested_after_backup"])
        self.assertFalse(receipt["stop_requested_after_download"])

    def test_successful_gcs_download_never_issues_client_side_stop(self):
        destination = self.root / "download"
        destination.mkdir()
        archive = destination / "results.tar.gz"
        create_result_archive(archive)
        marker = {"sha256": pipeline.sha256(archive), "compute_exit_code": 0,
                  "backup_verified": True, "stop_requested_after_backup": True}
        cloud = mock.Mock()
        cloud.call.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(marker))
        cloud.describe.return_value = {"status": "RUNNING"}
        state = {"gcs_prefix": "gs://test/rank", "run_id": "rank-test"}
        self.assertEqual(pipeline.gcs_download(cloud, state, destination), 0)
        cloud.describe.assert_not_called()
        cloud.stop.assert_not_called()
        receipt = json.loads((destination / "download.json").read_text())
        self.assertFalse(receipt["stop_requested_after_download"])

    def test_remote_shutdown_failure_updates_marker_without_retrying_stop(self):
        archive = self.root / "results.tar.gz"
        create_result_archive(archive)
        state = {"gcs_prefix": "gs://test/rank", "run_id": "rank-test", "project": "test",
                 "zone": "test-zone", "instance": "test-vm", "shutdown_method": "guest"}
        cloud = mock.Mock()
        process = mock.Mock()
        process.stdout = io.BytesIO(archive.read_bytes())
        process.wait.return_value = 0
        with mock.patch.object(pipeline, "Cloud", return_value=cloud), \
             mock.patch.object(pipeline.subprocess, "Popen", return_value=process), \
             mock.patch.object(pipeline, "command", side_effect=RuntimeError("sudo shutdown failed")) as shutdown:
            with self.assertRaisesRegex(RuntimeError, "sudo shutdown failed"):
                pipeline.upload_and_stop(self.root, state, archive, pipeline.sha256(archive), {"compute_exit_code": 0})
            shutdown.assert_called_once()
        marker = json.loads((self.root / "complete.json").read_text())
        self.assertFalse(marker["stop_requested_after_backup"])
        self.assertEqual(marker["stop_error"], "sudo shutdown failed")
        self.assertEqual(json.loads((self.root / "status.json").read_text())["state"], "stop_failed")
        self.assertEqual(cloud.call.call_count, 3)
        for call in cloud.call.call_args_list:
            self.assertEqual(call.args[0][:2], ["storage", "cp"])

    def test_gcs_pipeline_error_marker_never_describes_or_stops_vm(self):
        destination = self.root / "download"
        destination.mkdir()
        archive = destination / "results.tar.gz"
        create_result_archive(archive)
        marker = {"sha256": pipeline.sha256(archive), "compute_exit_code": 0,
                  "pipeline_error": "earlier backup error", "stop_requested_after_backup": False}
        cloud = mock.Mock()
        cloud.call.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(marker))
        state = {"gcs_prefix": "gs://test/rank", "run_id": "rank-test"}
        self.assertEqual(pipeline.gcs_download(cloud, state, destination), 1)
        cloud.describe.assert_not_called()
        cloud.stop.assert_not_called()

    def test_package_preserves_failed_status_and_partial_output(self):
        (self.root / "results").mkdir()
        (self.root / "results/partial.json").write_text('{"complete":false}')
        (self.root / "logs").mkdir()
        (self.root / "logs/pipeline.log").write_text("failed stage")
        pipeline.atomic_json(self.root / "status.json", {"state": "failed", "compute_exit_code": 2})
        archive, digest = pipeline.package_results(self.root)
        self.assertEqual(digest, pipeline.sha256(archive))
        with tarfile.open(archive) as result:
            self.assertIn("logs/pipeline.log", result.getnames())
            self.assertEqual(json.load(result.extractfile("status.json"))["compute_exit_code"], 2)


    def test_dry_run_never_invokes_cloud_or_subprocess(self):
        (self.root / "configs").mkdir()
        (self.root / "configs/recompute-plan.json").write_text('{"stages":[]}')
        (self.root / "tools").mkdir()
        (self.root / "tools/run_recompute_plan.py").write_text("pass\n")
        with mock.patch.object(pipeline, "ROOT", self.root), \
             mock.patch.object(pipeline, "Cloud") as cloud, \
             mock.patch.object(pipeline, "command") as external, \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(pipeline.local_main(["--dry-run"]), 0)
            cloud.assert_not_called()
            external.assert_not_called()

    def test_remote_failed_plan_still_archives_and_returns_nonzero(self):
        (self.root / "project").mkdir()
        (self.root / "venv/bin").mkdir(parents=True)
        (self.root / "venv/bin/python").write_text("placeholder")
        state = {"run_id": "rank-test", "plan": "configs/plan.json", "workers": 2}
        pipeline.atomic_json(self.root / "launch.json", state)
        with mock.patch.object(pipeline, "experiment_python", return_value=str(self.root / "venv/bin/python")), \
             mock.patch.object(pipeline.subprocess, "run", side_effect=[SimpleNamespace(returncode=0, stdout="Python 3.12"), SimpleNamespace(returncode=0), SimpleNamespace(returncode=0), SimpleNamespace(returncode=2)]) as run:
            self.assertEqual(pipeline.remote_main(self.root), 2)
        self.assertIn("run_recompute_plan.py", str(run.call_args_list[-1]))
        self.assertTrue((self.root / "results.tar.gz").is_file())
        status = json.loads((self.root / "status.json").read_text())
        self.assertEqual(status["state"], "archive_ready")
        self.assertEqual(status["compute_exit_code"], 2)

    def test_reuses_valid_experiment_environment(self):
        python = self.root / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("placeholder")
        with mock.patch.object(pipeline, "usable_python", return_value=True), \
             mock.patch.object(pipeline.subprocess, "run") as run:
            self.assertEqual(pipeline.experiment_python(self.root, io.StringIO(), {}), str(python))
        run.assert_not_called()

    def test_accepts_default_python310_without_interpreter_download(self):
        with mock.patch.object(pipeline.sys, "executable", "/usr/bin/python3.10"), \
             mock.patch.object(pipeline.shutil, "which", return_value=None), \
             mock.patch.object(pipeline, "usable_python", return_value=True), \
             mock.patch.object(pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            result = pipeline.experiment_python(self.root, io.StringIO(), {})
        self.assertEqual(result, str(self.root / "venv/bin/python"))
        self.assertEqual(run.call_args.args[0][:3], ["/usr/bin/python3.10", "-m", "venv"])
        self.assertEqual(run.call_count, 1)

    def test_missing_ensurepip_uses_bundled_offline_bootstrap(self):
        with mock.patch.object(pipeline.shutil, "which", return_value=None), \
             mock.patch.object(pipeline, "usable_python", side_effect=[True, False, True]), \
             mock.patch.object(pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            result = pipeline.experiment_python(self.root, io.StringIO(), {})
        self.assertEqual(result, str(self.root / "venv-bootstrap/bin/python"))
        argv = run.call_args.args[0]
        self.assertIn("--no-download", argv)
        self.assertIn("--no-periodic-update", argv)
        self.assertIn("virtualenv.pyz", argv[1])
        self.assertNotIn("uv", " ".join(argv))

    def test_failed_system_venv_uses_offline_fallback(self):
        with mock.patch.object(pipeline.shutil, "which", return_value=None), \
             mock.patch.object(pipeline, "usable_python", side_effect=[True, True, True]), \
             mock.patch.object(pipeline.subprocess, "run", side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as run:
            result = pipeline.experiment_python(self.root, io.StringIO(), {})
        self.assertEqual(result, str(self.root / "venv-bootstrap/bin/python"))
        self.assertEqual(run.call_count, 2)
        self.assertIn("--no-download", run.call_args.args[0])

    def test_runtime_failure_archives_logs_without_starting_experiment(self):
        (self.root / "project").mkdir()
        pipeline.atomic_json(self.root / "launch.json", {"run_id": "rank-test", "workers": 2})
        with mock.patch.object(pipeline, "experiment_python", side_effect=OSError("environment setup failed")), \
             mock.patch.object(pipeline.subprocess, "run") as run:
            self.assertEqual(pipeline.remote_main(self.root), 1)
        run.assert_not_called()
        self.assertTrue((self.root / "results.tar.gz").is_file())
        self.assertIn("environment setup failed", (self.root / "logs/pipeline.log").read_text())
        self.assertEqual(json.loads((self.root / "status.json").read_text())["state"], "archive_ready")

    def test_controller_source_parses_on_python38(self):
        import ast
        ast.parse(MODULE_PATH.read_text(), feature_version=(3, 8))

    def test_retry_cannot_start_with_stale_gcs_completion_marker(self):
        cloud = mock.Mock()
        cloud.ssh.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"state": "uploaded", "compute_exit_code": 2}))
        cloud.call.side_effect = [SimpleNamespace(returncode=0), RuntimeError("cannot remove stale marker")]
        state = {"remote_dir": "/run", "gcs_prefix": "gs://test/rank", "run_id": "rank-test"}
        with self.assertRaisesRegex(RuntimeError, "stale marker"):
            pipeline.launch_remote(cloud, state, Path("bundle"), Path("state"), retry=True)
        cloud.scp.assert_not_called()

if __name__ == "__main__":
    unittest.main()
