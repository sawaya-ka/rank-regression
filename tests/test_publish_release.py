import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("publish_release", Path(__file__).resolve().parents[1] / "tools/publish_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class PublicationTests(unittest.TestCase):
    def manifest(self, root, names):
        (root / "MANIFEST.sha256").write_text("".join(f"{release.digest(root / name)}  {name}\n" for name in names))

    def make_archive(self, path, records):
        with tarfile.open(path, "w:gz") as stream:
            for name, value in records:
                member = tarfile.TarInfo(name)
                member.size = len(value)
                stream.addfile(member, io.BytesIO(value))

    def test_archive_preserves_science_and_omits_operations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "original.tar.gz"
            records = [("results/checkpoints/0001.json", b'{"success":true}\n'),
                       ("results/parent_failures/0001.json", b'{"success":false}\n'),
                       ("source/repair_run.py", b"x = 1\n"),
                       ("launch.json", b'{"remote_dir":"/home/private"}'),
                       ("logs/pipeline.log", b"private VM name")]
            self.make_archive(original, records)
            first, second = root / "one.tar.gz", root / "two.tar.gz"
            result = release.build_asset(original, first, input_sha="a" * 64)
            release.build_asset(original, second, input_sha="a" * 64)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(result["files"], 3)
            with tarfile.open(first) as stream:
                for name, value in records[:3]:
                    self.assertEqual(stream.extractfile(name).read(), value)
                self.assertNotIn("launch.json", stream.getnames())
                manifest = json.load(stream.extractfile("PUBLIC_ARCHIVE_MANIFEST.json"))
                self.assertEqual(manifest["excluded_files"], ["launch.json", "logs/pipeline.log"])

    def test_unsafe_and_duplicate_members_rejected(self):
        for names in [["../escape"], ["/absolute"], ["results/a", "results/a"]]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                original = root / "original.tar.gz"
                self.make_archive(original, [(name, b"x") for name in names])
                with self.assertRaises(ValueError):
                    release.build_asset(original, root / "public.tar.gz")
                self.assertFalse((root / "public.tar.gz").exists())
                self.assertFalse((root / "public.tar.gz.partial").exists())

    def test_links_rejected_even_if_excluded(self):
        member = tarfile.TarInfo("logs/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        with self.assertRaises(ValueError):
            release.validate_member(member, set())

    def test_checksum_required(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "result"
            file.write_bytes(b"content")
            self.assertEqual(release.verify_input(file, release.digest(file), 7), release.digest(file))
            with self.assertRaises(ValueError):
                release.verify_input(file, "0" * 64, 7)

    def test_owner_guard(self):
        release.validate_repo("sawaya-ka/rank-repro", "sawaya-ka")
        for repo in ["other/rank-repro", "https://github.com/sawaya-ka/rank", "sawaya-ka/../../x"]:
            with self.subTest(repo=repo), self.assertRaises(ValueError):
                release.validate_repo(repo, "sawaya-ka")

    def test_large_git_file_rejected_without_loading_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with (root / "raw.bin").open("wb") as stream:
                stream.truncate(release.MAX_GIT_BYTES)
            with self.assertRaises(ValueError):
                release.validate_git_paths(root, ["raw.bin"])

    def test_manifest_checks_every_hash_and_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "README.md").write_text("reviewed")
            self.manifest(root, ["README.md"])
            self.assertEqual(release.manifest_paths(root), ["README.md", "MANIFEST.sha256"])
            (root / "README.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                release.manifest_paths(root)

    def test_manifest_rejects_traversal_and_parent_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["../x", "/x", "a/../x", "./x", ".git/config"]:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                    release.validate_git_paths(root, [name])
            (root / "actual").mkdir()
            (root / "actual/file").write_text("reviewed")
            (root / "link").symlink_to(root / "actual", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                release.validate_git_paths(root, ["link/file"])

    def test_extra_staged_or_untracked_files_rejected(self):
        for name in [".env.local", "credentials.json", "notes.txt"]:
            with self.subTest(name=name), patch.object(release, "command", return_value=SimpleNamespace(stdout="README.md\0" + name + "\0")):
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    release.check_git_selection(Path("."), ["README.md", "MANIFEST.sha256"])

    def test_only_manifest_files_are_staged(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "GIT_AUTHOR_NAME": "Publication Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Publication Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }):
            root = Path(folder)
            (root / ".gitignore").write_text("ignored.txt\n")
            (root / "README.md").write_text("reviewed")
            (root / "ignored.txt").write_text("not for publication")
            self.manifest(root, [".gitignore", "README.md"])
            release.prepare_git(root, "owner/repo")
            tracked = release.command(["git", "ls-files"], cwd=root).stdout.splitlines()
            self.assertEqual(tracked, [".gitignore", "MANIFEST.sha256", "README.md"])

    def test_release_different_tag_commit_is_rejected(self):
        with patch.object(release, "api", return_value={"object": {"type": "commit", "sha": "b" * 40}}):
            with self.assertRaisesRegex(ValueError, "tag points"):
                release.verify_release_target("owner/repo", "a" * 40)

    def test_release_annotated_tag_is_resolved(self):
        with patch.object(release, "api", side_effect=[
            {"object": {"type": "tag", "sha": "b" * 40}},
            {"object": {"type": "commit", "sha": "a" * 40}},
        ]):
            release.verify_release_target("owner/repo", "a" * 40)

    def test_release_wrong_draft_target_blocks_publication(self):
        draft = {"draft": True, "tag_name": release.TAG, "target_commitish": "other-branch"}
        with patch.object(release, "api", side_effect=[None, {"sha": "b" * 40}]):
            with self.assertRaisesRegex(ValueError, "target resolves"):
                release.verify_release_target("owner/repo", "a" * 40, draft)

    def test_release_correct_draft_target_branch_resolves(self):
        draft = {"draft": True, "tag_name": release.TAG, "target_commitish": "main"}
        with patch.object(release, "api", side_effect=[None, {"sha": "a" * 40}]):
            release.verify_release_target("owner/repo", "a" * 40, draft)

    def test_existing_repository_rejected_before_local_git_changes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(release.shutil, "which", return_value="/bin/tool"), patch.object(release, "api", side_effect=[{"login": "owner"}, {"id": 1}]), patch.object(release, "prepare_git") as prepare:
            with self.assertRaisesRegex(ValueError, "already exists"):
                release.publish(Path(folder), "owner/repo", [], Path("notes.md"))
            prepare.assert_not_called()

    def test_remote_asset_digest_mismatch_blocks_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "asset.tar.gz"
            path.write_bytes(b"abc")
            response = {"assets": [{"name": path.name, "size": 3, "digest": "sha256:" + "0" * 64}]}
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                release.verify_remote_assets("owner/repo", response, [path])

    def test_failed_upload_never_publishes_draft(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "raw.tar.gz"
            path.write_bytes(b"abc")
            draft = {"id": 3, "tag_name": release.TAG, "draft": True, "assets": []}
            commands = []

            def call(argv, **kwargs):
                commands.append(argv)
                if argv[:3] == ["gh", "release", "upload"]:
                    raise RuntimeError("connection lost")

            with patch.object(release.shutil, "which", return_value="/bin/tool"), \
                 patch.object(release, "api", side_effect=[{"login": "owner"}, None, [], [draft]]), \
                 patch.object(release, "prepare_git", return_value=(root / "state.json", {"repo": "owner/repo", "commit": "a" * 40})), \
                 patch.object(release, "verify_release_target"), \
                 patch.object(release, "command", side_effect=call):
                with self.assertRaisesRegex(RuntimeError, "connection lost"):
                    release.publish(root, "owner/repo", [path], root / "notes.md")
            self.assertFalse(any(command[:3] == ["gh", "release", "edit"] for command in commands))
            self.assertFalse(any("--clobber" in command or "--force" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
