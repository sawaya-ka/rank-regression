#!/usr/bin/env python3
"""Run an experiment plan on an existing GCP VM; collect before stopping.

No cloud operation occurs at import time or with --dry-run.
The local client requires only Python >= 3.9 and an authenticated gcloud CLI.
The VM controller needs Python >= 3.8. Experiments use an installed Python
3.10--3.12 in an isolated environment; no interpreter installer is downloaded.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tools", "configs", "data")
SOURCE_FILES = ("pyproject.toml", "requirements-tested.txt", "README.md", "run_gcp.sh")
SKIP_PARTS = {"__pycache__", ".venv", "checkpoints", "smoke", "runs", "cache", ".git"}
TERMINAL = {"archive_ready", "uploaded", "upload_failed", "stop_failed"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(argv, *, check=True, capture=True, **kwargs):
    result = subprocess.run([str(x) for x in argv], text=True,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, **kwargs)
    if check and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(map(str, argv))}\n{result.stderr or ''}")
    return result


def source_files(root):
    files = [root / name for name in SOURCE_FILES if (root / name).is_file()]
    for name in SOURCE_DIRS:
        for path in (root / name).rglob("*"):
            relative = path.relative_to(root)
            if any(part.startswith(".") or part in SKIP_PARTS or part.endswith(".egg-info") for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Source bundle does not follow symlinks: {relative}")
            if path.is_file() and path.suffix not in {".pyc", ".zip", ".gz", ".pem", ".key"}:
                files.append(path)
    return sorted(set(files))


def create_bundle(root, destination):
    """Create a deterministic, explicitly scoped source/data snapshot."""
    files = source_files(root)
    records = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256(path)} for path in files]
    manifest = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in files:
                info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
            info = tarfile.TarInfo("source-manifest.json")
            info.size = len(manifest)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(manifest))
    return digest, len(files), sum(item["bytes"] for item in records)


def safe_extract(archive_path, destination):
    """Extract only ordinary files/directories under the chosen destination."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive escapes destination: {member.name}")
        for member in archive.getmembers():
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def validate_hash(archive, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Malformed SHA-256 checksum")
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(f"Checksum mismatch: expected {expected}, received {actual}; VM will not be stopped")


class Cloud:
    def __init__(self, project, zone=None, instance=None, ssh_user=None, iap=False):
        self.project, self.zone, self.instance = project, zone, instance
        self.ssh_user, self.iap = ssh_user, iap

    def call(self, args, **kwargs):
        return command(["gcloud", *args, "--project", self.project, "--quiet"], **kwargs)

    def describe(self):
        return json.loads(self.call(["compute", "instances", "describe", self.instance,
                                     "--zone", self.zone, "--format=json"]).stdout)

    @property
    def target(self):
        return f"{self.ssh_user}@{self.instance}" if self.ssh_user else self.instance

    def ssh(self, script, **kwargs):
        args = ["compute", "ssh", self.target, "--zone", self.zone, "--command", script]
        if self.iap:
            args.append("--tunnel-through-iap")
        return self.call(args, **kwargs)

    def scp(self, source, destination, **kwargs):
        args = ["compute", "scp", str(source), str(destination), "--zone", self.zone]
        if self.iap:
            args.append("--tunnel-through-iap")
        return self.call(args, **kwargs)

    def stop(self):
        self.call(["compute", "instances", "stop", self.instance, "--zone", self.zone], capture=False)

    def choose(self):
        rows = json.loads(self.call(["compute", "instances", "list", "--format=json"]).stdout)
        rows = [row for row in rows if row.get("status") in {"RUNNING", "TERMINATED"}]
        if not rows:
            raise ValueError("No running or stopped VM found; this launcher does not create VMs")
        print("Select the COMPUTE VM that may be stopped after successful completion (no VM is selected by default):")
        for index, row in enumerate(rows, 1):
            print(f"{index:2d}. {row['name']}  {row['zone'].split('/')[-1]}  "
                  f"{row['machineType'].split('/')[-1]}  {row['status']}")
        if not sys.stdin.isatty():
            raise ValueError("Noninteractive use requires --instance and --zone")
        answer = input("VM number: ").strip()
        if not answer.isdigit() or not 1 <= int(answer) <= len(rows):
            raise ValueError("No valid VM selected")
        row = rows[int(answer) - 1]
        self.instance, self.zone = row["name"], row["zone"].split("/")[-1]


def verified_download_then_stop(cloud, state, destination, status):
    """Stop only successful runs after download + independent SHA-256.

    Failed runs still return their verified logs/results, but retain the VM.
    A prior backup or shutdown error is not silently retried as a stop here.
    """
    destination.mkdir(parents=True, exist_ok=True)
    remote = state["remote_dir"]
    checksum_text = cloud.ssh("cat " + shlex.quote(remote + "/results.sha256")).stdout.split()
    if not checksum_text:
        raise ValueError("Remote checksum is missing; VM will not be stopped")
    checksum = checksum_text[0]
    archive = destination / "results.tar.gz"
    if not archive.is_file() or sha256(archive) != checksum:
        partial = destination / "results.tar.gz.partial"
        cloud.scp(f"{cloud.target}:{remote}/results.tar.gz", partial, capture=False)
        validate_hash(partial, checksum)
        partial.replace(archive)
    validate_hash(archive, checksum)
    (destination / "results.sha256").write_text(checksum + "  results.tar.gz\n")
    safe_extract(archive, destination / "extracted")
    exit_code = int(status["compute_exit_code"])
    previous_error = status.get("state") in {
        "failed", "upload_failed", "stop_failed", "packaging_failed", "worker_interrupted"
    }
    may_stop = exit_code == 0 and not previous_error
    atomic_json(destination / "download.json", {"sha256": checksum, "verified": True,
                "compute_exit_code": exit_code, "instance": cloud.instance,
                "stop_requested_after_download": may_stop})
    print(f"Verified download: {archive}", flush=True)
    if not may_stop:
        print("Run or pipeline failed. Verified logs/results were downloaded; "
              "the VM is being kept running and no stop was requested.", flush=True)
        return exit_code or 1
    cloud.stop()
    return 0


def gcs_download(cloud, state, destination):
    """A marker is published only after the VM reads back and hashes its upload."""
    prefix = state["gcs_prefix"].rstrip("/") + "/" + state["run_id"]
    result = cloud.call(["storage", "cat", prefix + "/complete.json"], check=False)
    if result.returncode:
        return None
    marker = json.loads(result.stdout)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "results.tar.gz"
    if not archive.is_file() or sha256(archive) != marker["sha256"]:
        partial = destination / "results.tar.gz.partial"
        cloud.call(["storage", "cp", prefix + "/results.tar.gz", str(partial)], capture=False)
        validate_hash(partial, marker["sha256"])
        partial.replace(archive)
    validate_hash(archive, marker["sha256"])
    (destination / "results.sha256").write_text(marker["sha256"] + "  results.tar.gz\n")
    safe_extract(archive, destination / "extracted")
    exit_code = int(marker["compute_exit_code"])
    previous_error = bool(marker.get("pipeline_error") or marker.get("stop_error"))
    may_stop = exit_code == 0 and not previous_error
    # Old failed-run markers incorrectly requested a stop. The exit status is
    # authoritative; never propagate that obsolete flag into a new receipt.
    receipt = {**marker, "verified": True, "stop_requested_after_download": False}
    if not may_stop:
        receipt["stop_requested_after_backup"] = False
    atomic_json(destination / "download.json", receipt)
    print(f"Verified GCS download: {archive}", flush=True)
    if not may_stop:
        print("Run or pipeline failed. Verified logs/results were downloaded from GCS; "
              "the VM is being kept running and no stop was requested.", flush=True)
        return exit_code or 1
    # Only the remote worker may stop a successful GCS-mode run. A client-side
    # fallback could hide a failed remote shutdown or race with that error.
    print("Automatic VM stop is handled by the remote worker after successful computation; "
          "the download client will not issue a stop command.", flush=True)
    return 0


def launch_remote(cloud, state, local_bundle, local_state, retry=False):
    remote = state["remote_dir"]
    previous = cloud.ssh("cat " + shlex.quote(remote + "/status.json"), check=False)
    if previous.returncode == 0:
        status = json.loads(previous.stdout)
        if status.get("state") in TERMINAL and (not retry or status.get("compute_exit_code") == 0):
            print("Existing archived run found; collecting without rerunning", flush=True)
            return
        if status.get("state") == "running":
            alive = cloud.ssh("kill -0 " + str(int(status["pid"])), check=False)
            if alive.returncode == 0:
                print("Worker already running; reconnecting", flush=True)
                return
    if retry and state.get("gcs_prefix"):
        marker = state["gcs_prefix"].rstrip("/") + "/" + state["run_id"] + "/complete.json"
        exists = cloud.call(["storage", "cat", marker], check=False)
        if exists.returncode == 0:
            cloud.call(["storage", "rm", marker])
    cloud.ssh("mkdir -p " + shlex.quote(remote))
    cloud.scp(local_bundle, f"{cloud.target}:{remote}/source.tar.gz", capture=False)
    cloud.scp(local_state, f"{cloud.target}:{remote}/launch.json", capture=False)
    script = "\n".join([
        "set -eu", "cd " + shlex.quote(remote),
        "printf '%s  source.tar.gz\\n' " + shlex.quote(state["bundle_sha256"]) + " | sha256sum -c -",
        "mkdir -p project", "tar -xzf source.tar.gz -C project",
        "printf '{\"state\":\"queued\"}\\n' > status.json",
        # flock prevents simultaneous duplicate workers from two local clients.
        "nohup setsid flock -n worker.lock bash project/tools/remote_pipeline.sh "
        + shlex.quote(remote) + " > launcher.log 2>&1 < /dev/null &",
    ])
    cloud.ssh(script)


def watch(cloud, state, destination, interval):
    print(f"Run ID: {state['run_id']}\nRemote directory: {state['remote_dir']}", flush=True)
    print(f"Reconnect with: ./run_gcp.sh --resume {state['run_id']}", flush=True)
    if not state.get("gcs_prefix"):
        print("Direct mode: computation survives disconnects; download and automatic stop require this client to reconnect.", flush=True)
    previous = None
    while True:
        try:
            if state.get("gcs_prefix"):
                complete = gcs_download(cloud, state, destination)
                if complete is not None:
                    return complete
            status_script = "\n".join([
                "import json, os, pathlib",
                "s = json.loads(pathlib.Path(" + repr(state["remote_dir"] + "/status.json") + ").read_text())",
                "if s.get('state') == 'running':",
                "    try: os.kill(int(s['pid']), 0)",
                "    except ProcessLookupError: s['state'] = 'worker_interrupted'",
                "print(json.dumps(s))",
            ])
            result = cloud.ssh("python3 -c " + shlex.quote(status_script), check=False)
            status = json.loads(result.stdout) if result.returncode == 0 else {"state": "starting_or_unreachable"}
            description = status.get("state", "unknown")
            if description != previous:
                print(description + (": " + status["message"] if status.get("message") else ""), flush=True)
                previous = description
            if description == "worker_interrupted":
                print("The remote worker ended unexpectedly. Checkpoints remain intact.", flush=True)
                print(f"Restart with: ./run_gcp.sh --resume {state['run_id']} --retry-failed", flush=True)
                return 2
            if description == "packaging_failed":
                print("Packaging failed. Files remain on the VM; no stop was issued. See launcher.log.", flush=True)
                return 2
            if description in TERMINAL and not state.get("gcs_prefix"):
                return verified_download_then_stop(cloud, state, destination, status)
            if description in {"upload_failed", "stop_failed"}:
                # SSH may still work, and direct transfer is a safe recovery path.
                print("Remote backup/stop failed; attempting verified direct download before stopping.", flush=True)
                return verified_download_then_stop(cloud, state, destination, status)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"Connection/transfer incomplete: {exc}\nRetrying; no unverified-data stop will be issued.", flush=True)
        time.sleep(interval)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None)
    parser.add_argument("--zone")
    parser.add_argument("--instance", help="Existing dedicated compute VM; omitted prompts for selection")
    parser.add_argument("--ssh-user")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--plan", default="configs/recompute-plan.json")
    parser.add_argument("--gcs-prefix", help="Existing gs://bucket/prefix; enables offline-safe backup and guest stop")
    parser.add_argument("--shutdown-method", choices=("guest", "gcloud"), default="guest")
    parser.add_argument("--download-dir", type=Path, default=ROOT / "downloads")
    parser.add_argument("--resume", help="Run ID printed by the first invocation")
    parser.add_argument("--retry-failed", action="store_true", help="Resume unfinished grid jobs after a failed archived run")
    parser.add_argument("--tunnel-through-iap", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Validate plan and bundle locally; never contact GCP")
    return parser.parse_args(argv)


def local_main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("--poll-seconds must lie between 1 and 60")
    state_root = ROOT / ".gcp-runs"
    if args.resume:
        if not re.fullmatch(r"rank-[a-f0-9]{16}", args.resume):
            raise ValueError("Invalid run ID")
        local_state = state_root / args.resume / "launch.json"
        state = json.loads(local_state.read_text())
        cloud = Cloud(state["project"], state["zone"], state["instance"], state.get("ssh_user"), state.get("iap", False))
        if args.dry_run:
            print(json.dumps(state, indent=2))
            return 0
        if state.get("gcs_prefix"):
            completed = gcs_download(cloud, state, args.download_dir / state["run_id"])
            if completed is not None and (not args.retry_failed or completed == 0):
                return completed
        info = cloud.describe()
        if info["status"] == "TERMINATED":
            downloaded = args.download_dir / state["run_id"]
            receipt = downloaded / "download.json"
            if receipt.is_file() and not args.retry_failed:
                record = json.loads(receipt.read_text())
                validate_hash(downloaded / "results.tar.gz", record["sha256"])
                print(f"Already downloaded and verified: {downloaded}")
                return int(record["compute_exit_code"])
            cloud.call(["compute", "instances", "start", cloud.instance, "--zone", cloud.zone], capture=False)
        if args.retry_failed:
            launch_remote(cloud, state, local_state.parent / "source.tar.gz", local_state, retry=True)
        return watch(cloud, state, args.download_dir / state["run_id"], args.poll_seconds)
    plan = (ROOT / args.plan).resolve()
    if not plan.is_relative_to(ROOT) or not plan.is_file():
        raise ValueError("--plan must be an existing JSON file inside this project")
    json.loads(plan.read_text())
    if not (ROOT / "tools/run_recompute_plan.py").is_file():
        raise ValueError("Missing tools/run_recompute_plan.py")
    staging = state_root / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    bundle = staging / "source.tar.gz"
    source_hash, count, size = create_bundle(ROOT, bundle)
    print(f"Source snapshot: {count} files, {size / 1024**2:.1f} MiB uncompressed", flush=True)
    if args.dry_run:
        print(json.dumps({"plan": str(plan.relative_to(ROOT)), "workers": args.workers,
              "source_sha256": source_hash, "bundle_sha256": sha256(bundle),
              "mode": "successful computation + verified GCS backup then stop" if args.gcs_prefix else "successful computation + verified local download then stop",
              "cloud_operations": 0}, indent=2))
        return 0
    if not shutil.which("gcloud"):
        raise ValueError("Install/authenticate the Google Cloud CLI before running this command")
    if args.gcs_prefix and not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[A-Za-z0-9_./-]+)?", args.gcs_prefix):
        raise ValueError("--gcs-prefix must be an existing gs://bucket/optional-prefix")
    project = args.project
    if not project:
        project = command(["gcloud", "config", "get-value", "project"]).stdout.strip()
    if not project or project == "(unset)":
        raise ValueError("Set --project PROJECT_ID or configure a project with gcloud config set project PROJECT_ID")
    cloud = Cloud(project, args.zone, args.instance, args.ssh_user, args.tunnel_through_iap)
    if not args.instance:
        cloud.choose()
    elif not args.zone:
        raise ValueError("--zone is required with --instance")
    info = cloud.describe()
    if any(disk.get("type") == "SCRATCH" for disk in info.get("disks", [])):
        raise ValueError("This launcher refuses automatic stop on a VM with Local SSD disks; choose a persistent-disk compute VM")
    machine = info["machineType"].split("/")[-1]
    cpus = json.loads(cloud.call(["compute", "machine-types", "describe", machine, "--zone", cloud.zone, "--format=json"]).stdout)["guestCpus"]
    workers = args.workers or max(1, int(cpus) - 2)
    if workers < 1 or workers > int(cpus):
        raise ValueError(f"--workers must lie between 1 and the VM's {cpus} vCPUs")
    if info.get("status") == "TERMINATED":
        cloud.call(["compute", "instances", "start", cloud.instance, "--zone", cloud.zone], capture=False)
    elif info.get("status") != "RUNNING":
        raise ValueError(f"VM is not ready: {info.get('status')}")
    remote_home = cloud.ssh("pwd").stdout.strip()
    if not remote_home.startswith("/") or "\n" in remote_home:
        raise ValueError("Could not determine the remote home directory")
    fingerprint = json.dumps([source_hash, str(plan.relative_to(ROOT)), workers, cloud.project,
                              cloud.zone, cloud.instance, args.gcs_prefix, args.shutdown_method]).encode()
    run_id = "rank-" + hashlib.sha256(fingerprint).hexdigest()[:16]
    directory = state_root / run_id
    directory.mkdir(exist_ok=True)
    shutil.copy2(bundle, directory / "source.tar.gz")
    state = {"run_id": run_id, "project": cloud.project, "zone": cloud.zone,
             "instance": cloud.instance, "ssh_user": cloud.ssh_user, "iap": cloud.iap,
             "workers": workers, "plan": str(plan.relative_to(ROOT)), "gcs_prefix": args.gcs_prefix,
             "shutdown_method": args.shutdown_method, "source_sha256": source_hash,
             "bundle_sha256": sha256(bundle), "remote_dir": remote_home + "/rank-repro-runs/" + run_id}
    local_state = directory / "launch.json"
    atomic_json(local_state, state)
    # These tests happen before the long experiment, and do not alter IAM.
    preflight = "command -v python3 >/dev/null && command -v flock >/dev/null && command -v setsid >/dev/null && python3 -c 'import sys; assert sys.version_info >= (3,8), \"VM controller needs Python 3.8+; experiment Python is prepared separately\"'"
    if args.gcs_prefix:
        preflight += " && command -v gcloud >/dev/null"
        if args.shutdown_method == "guest":
            preflight += " && sudo -n true"
    cloud.ssh(preflight)
    launch_remote(cloud, state, directory / "source.tar.gz", local_state)
    return watch(cloud, state, args.download_dir / run_id, args.poll_seconds)


def package_results(run_dir):
    """Capture partial output and logs on failure as well as successful results."""
    destination = run_dir / "results.tar.gz"
    temporary = destination.with_suffix(".gz.tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for name in ("results", "logs", "launcher.log", "status.json", "launch.json", "runtime.json"):
            path = run_dir / name
            if path.exists():
                archive.add(path, arcname=name)
        for name in ("source-manifest.json",):
            path = run_dir / "project" / name
            if path.exists():
                archive.add(path, arcname=name)
    temporary.replace(destination)
    digest = sha256(destination)
    (run_dir / "results.sha256").write_text(digest + "  results.tar.gz\n")
    return destination, digest


def upload_and_stop(run_dir, state, archive, digest, status):
    prefix = state["gcs_prefix"].rstrip("/") + "/" + state["run_id"]
    cloud = Cloud(state["project"], state["zone"], state["instance"])
    cloud.call(["storage", "cp", str(archive), prefix + "/results.tar.gz"], capture=False)
    # Read back the complete object and hash it without allocating its size in RAM.
    process = subprocess.Popen(["gcloud", "storage", "cat", prefix + "/results.tar.gz",
                                "--project", state["project"], "--quiet"], stdout=subprocess.PIPE)
    received = hashlib.sha256()
    for block in iter(lambda: process.stdout.read(4 * 1024 * 1024), b""):
        received.update(block)
    if process.wait() != 0 or received.hexdigest() != digest:
        raise RuntimeError("GCS read-back verification failed; refusing to stop the VM")
    exit_code = int(status["compute_exit_code"])
    previous_error = status.get("state") in {
        "failed", "upload_failed", "stop_failed", "packaging_failed", "worker_interrupted"
    }
    may_stop = exit_code == 0 and not previous_error
    marker = {"sha256": digest, "bytes": archive.stat().st_size,
              "compute_exit_code": exit_code, "run_id": state["run_id"],
              "backup_verified": True, "stop_requested_after_backup": may_stop}
    if previous_error:
        marker["pipeline_error"] = status.get("message") or status["state"]
    atomic_json(run_dir / "complete.json", marker)
    cloud.call(["storage", "cp", str(run_dir / "complete.json"), prefix + "/complete.json"], capture=False)
    message = ("GCS archive read-back verified; successful run, requesting VM stop" if may_stop
               else "Run or pipeline failed; GCS backup verified, keeping the VM running without requesting stop")
    status.update(state="uploaded", message=message)
    atomic_json(run_dir / "status.json", status)
    if not may_stop:
        print(message, flush=True)
        return
    try:
        if state["shutdown_method"] == "guest":
            command(["sudo", "-n", "shutdown", "-h", "now"], capture=False)
        else:
            cloud.call(["compute", "instances", "stop", state["instance"], "--zone", state["zone"], "--async"], capture=False)
    except Exception as exc:
        status.update(state="stop_failed", message=str(exc))
        atomic_json(run_dir / "status.json", status)
        marker.update(stop_requested_after_backup=False, stop_error=str(exc))
        atomic_json(run_dir / "complete.json", marker)
        try:
            cloud.call(["storage", "cp", str(run_dir / "complete.json"), prefix + "/complete.json"], capture=False)
        except Exception as marker_error:
            print(f"Could not publish the shutdown error to GCS; see VM status.json: {marker_error}", flush=True)
        raise


def usable_python(path, *, require_pip=False, require_venv=False):
    """Probe a concrete interpreter without relying on shell activation/PATH."""
    imports = "import sys"
    if require_pip:
        imports += "; import pip"
    if require_venv:
        imports += "; import venv, ensurepip"
    script = imports + "; assert (3,10) <= sys.version_info[:2] <= (3,12)"
    try:
        return subprocess.run([str(path), "-c", script], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def experiment_python(run_dir, log, environment):
    """Use an installed Python and bundled bootstrap; never download an interpreter."""
    for name in ("venv", "venv-bootstrap", "venv-managed"):
        candidate = run_dir / name / "bin/python"
        if candidate.is_file() and usable_python(candidate, require_pip=True):
            log.write("Reusing experiment Python: " + str(candidate) + "\n")
            return str(candidate)
    candidates = [sys.executable, shutil.which("python3.10"),
                  shutil.which("python3.12"), shutil.which("python3.11")]
    candidates = list(dict.fromkeys(value for value in candidates if value))
    candidate = next((path for path in candidates if usable_python(path)), None)
    if candidate is None:
        raise RuntimeError("No installed Python 3.10--3.12 found; no system changes were made.")
    if usable_python(candidate, require_venv=True):
        log.write("Creating isolated environment with " + candidate + "\n")
        result = subprocess.run([candidate, "-m", "venv", str(run_dir / "venv")],
                                stdout=log, stderr=subprocess.STDOUT, env=environment)
        python = run_dir / "venv/bin/python"
        if result.returncode == 0 and usable_python(python, require_pip=True):
            return str(python)
        log.write("System venv creation failed; using the bundled offline bootstrap.\n")
    bootstrap = ROOT / "tools/vendor/virtualenv.pyz"
    provenance = json.loads(bootstrap.with_name("virtualenv-provenance.json").read_text())
    validate_hash(bootstrap, provenance["sha256"])
    log.write("Preparing pip with bundled virtualenv; no bootstrap download is required.\n")
    subprocess.run([candidate, str(bootstrap), "--python", candidate,
                    "--no-download", "--no-periodic-update", "--pip", "embed", "--setuptools", "embed",
                    "--app-data", str(run_dir / "virtualenv-app-data"), str(run_dir / "venv-bootstrap")],
                   check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
    python = run_dir / "venv-bootstrap/bin/python"
    if not usable_python(python, require_pip=True):
        raise RuntimeError("Bundled bootstrap did not produce a usable environment with pip; see pipeline.log")
    return str(python)


def remote_main(run_directory):
    run_dir = Path(run_directory).resolve()
    state = json.loads((run_dir / "launch.json").read_text())
    status_path = run_dir / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get("compute_exit_code") == 0 and (run_dir / "results.tar.gz").is_file():
        print("Successful archive already exists; not rerunning computations", flush=True)
        return 0
    (run_dir / "logs").mkdir(exist_ok=True)
    status = {"state": "running", "compute_exit_code": None, "run_id": state["run_id"],
              "pid": os.getpid(), "started_unix": time.time()}
    atomic_json(status_path, status)
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    environment.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    project = run_dir / "project"
    code = 1
    with (run_dir / "logs/pipeline.log").open("a", buffering=1) as log:
        try:
            if state.get("gcs_prefix"):
                prefix = state["gcs_prefix"].rstrip("/") + "/" + state["run_id"]
                cloud = Cloud(state["project"])
                # Test VM service-account write + read rights before costly work.
                probe = run_dir / "gcs-probe.txt"
                probe.write_text(state["run_id"])
                cloud.call(["storage", "cp", str(probe), prefix + "/probe.txt"])
                received = cloud.call(["storage", "cat", prefix + "/probe.txt"]).stdout
                if received != state["run_id"]:
                    raise RuntimeError("GCS probe verification failed")
            python = experiment_python(run_dir, log, environment)
            version = subprocess.run([python, "-VV"], check=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True).stdout.strip()
            atomic_json(run_dir / "runtime.json", {"experiment_python": python, "version": version,
                                                   "controller_python": sys.version})
            subprocess.run([python, "-m", "pip", "install", "-r", str(project / "requirements-tested.txt"), "-e", str(project)],
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
            with (run_dir / "logs/environment.txt").open("w") as installed:
                subprocess.run([python, "-m", "pip", "freeze"], check=True, stdout=installed,
                               stderr=log, env=environment)
            # The experiment runner handles computational stages and resume checks.
            argv = [python, str(project / "tools/run_recompute_plan.py"), "--plan", state["plan"],
                    "--output-dir", str(run_dir / "results"), "--workers", str(state["workers"])]
            log.write("Command: " + shlex.join(argv) + "\n")
            result = subprocess.run(argv, cwd=project, stdout=log, stderr=subprocess.STDOUT, env=environment)
            code = result.returncode
        except Exception:
            traceback.print_exc(file=log)
            code = 1
    status.update(state="succeeded" if code == 0 else "failed", compute_exit_code=code, finished_unix=time.time())
    atomic_json(status_path, status)
    try:
        archive, digest = package_results(run_dir)
    except Exception as exc:
        status.update(state="packaging_failed", message=str(exc))
        atomic_json(status_path, status)
        raise
    status.update(state="archive_ready", sha256=digest)
    atomic_json(status_path, status)
    if state.get("gcs_prefix"):
        try:
            upload_and_stop(run_dir, state, archive, digest, status)
        except Exception as exc:
            if status.get("state") != "stop_failed":
                status.update(state="upload_failed", message=str(exc))
                atomic_json(status_path, status)
            print(f"Backup or stop failed; retained data on persistent disk: {exc}", flush=True)
            return 1
    return code


def main():
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "_worker":
            return remote_main(sys.argv[2])
        return local_main()
    except KeyboardInterrupt:
        print("\nLocal monitoring stopped. Detached VM computation continues. Use the printed --resume command.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"rank-gcp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
