#!/usr/bin/env python3
"""Prepare the reviewed raw archive and optionally publish a new personal GitHub repository.

The default is an offline preview. --build writes release assets; --publish also
creates a public repository and release. This command never contacts GCP.
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
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA256 = "9186dbf80fdfe6b25b67a72f05d026015a3ab022d3ed023c0c925d2db2bc48f4"
INPUT_BYTES = 546645073
TAG = "v0.3.0"
ASSET_NAME = "rank-results-v0.3.0.tar.gz"
MAX_GIT_BYTES = 100 * 1024 * 1024
PUBLIC_PREFIXES = ("results/", "source/")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_input(path, expected_sha=INPUT_SHA256, expected_bytes=INPUT_BYTES):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"Expected the reviewed {expected_bytes}-byte results.tar.gz")
    actual = digest(path)
    if actual != expected_sha:
        raise ValueError(f"Input SHA-256 mismatch: expected {expected_sha}, received {actual}")
    return actual


def validate_member(member, seen):
    name = member.name
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or not (member.isfile() or member.isdir())):
        raise ValueError(f"Unsafe archive member: {name}")
    if name in seen:
        raise ValueError(f"Duplicate archive member: {name}")
    seen.add(name)
    return member.isfile() and name.startswith(PUBLIC_PREFIXES)


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.hash.update(block)
        return block


def build_asset(source, destination, *, input_sha=INPUT_SHA256):
    """Stream bytes unchanged into a deterministic archive, omitting operational logs."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    records, excluded, seen = [], [], set()
    try:
        with tarfile.open(source, "r|gz") as original, temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as public:
                    for member in original:
                        keep = validate_member(member, seen)
                        if not member.isfile():
                            continue
                        if not keep:
                            excluded.append(member.name)
                            continue
                        reader = HashReader(original.extractfile(member))
                        info = tarfile.TarInfo(member.name)
                        info.size, info.mode = member.size, 0o644
                        public.addfile(info, reader)
                        records.append({"path": member.name, "bytes": member.size,
                                        "sha256": reader.hash.hexdigest()})
                    manifest = {
                        "format": "rank-public-results-v1",
                        "original_archive_sha256": input_sha,
                        "policy": "Retain every regular file under results/ and source/ byte-for-byte; omit cloud launch/status records and logs. Normalize tar ownership, modes, and timestamps only.",
                        "excluded_files": excluded,
                        "files": records,
                    }
                    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
                    info = tarfile.TarInfo("PUBLIC_ARCHIVE_MANIFEST.json")
                    info.size, info.mode = len(data), 0o644
                    public.addfile(info, io.BytesIO(data))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"name": destination.name, "bytes": destination.stat().st_size,
            "sha256": digest(destination), "files": len(records),
            "excluded_files": excluded, "original_archive_sha256": input_sha}


def command(argv, *, cwd=ROOT, check=True):
    environment = dict(os.environ, GH_HOST="github.com")
    completed = subprocess.run([str(value) for value in argv], cwd=cwd, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    if check and completed.returncode:
        raise RuntimeError(f"{argv[0]} {' '.join(map(str, argv[1:3]))} failed: {completed.stderr.strip()}")
    return completed


def api(endpoint):
    result = command(["gh", "api", endpoint], check=False)
    if result.returncode:
        if "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(result.stderr.strip() or "GitHub request failed")
    return json.loads(result.stdout)


def find_release(repo):
    # List releases because the get-by-tag endpoint is intended for published
    # releases; authenticated owners can see their drafts in this listing.
    page = 1
    while True:
        entries = api(f"repos/{repo}/releases?per_page=100&page={page}")
        if entries is None:
            raise RuntimeError("Cannot list releases in the selected repository")
        matches = [item for item in entries if item["tag_name"] == TAG]
        if len(matches) > 1:
            raise ValueError("Multiple releases use the selected tag; inspect them manually")
        if matches:
            return matches[0]
        if len(entries) < 100:
            return None
        page += 1


def validate_repo(repo, login):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._-]+", repo):
        raise ValueError("--repo must be OWNER/NAME on github.com")
    if repo.split("/", 1)[1] in {".", ".."}:
        raise ValueError("Invalid repository name")
    if repo.split("/", 1)[0].lower() != login.lower():
        raise ValueError("Repository owner must match the authenticated GitHub account; this helper only creates personal repositories")


def validate_git_paths(root, names):
    for name in names:
        lexical = PurePosixPath(name)
        if (lexical.is_absolute() or ".." in lexical.parts or "\\" in name
                or lexical.as_posix() != name or ".git" in lexical.parts
                or any(ord(character) < 32 for character in name)):
            raise ValueError(f"Unsafe Git manifest path: {name!r}")
        path = root / name
        if any((root.joinpath(*lexical.parts[:index])).is_symlink()
               for index in range(1, len(lexical.parts) + 1)):
            raise ValueError(f"Refusing to publish a symlink: {name}")
        if not path.is_file():
            raise ValueError(f"Manifest file missing or not regular: {name}")
        if path.is_file() and path.stat().st_size >= MAX_GIT_BYTES:
            raise ValueError(f"File is too large for Git; use a release asset: {name}")
        if path.name.startswith(".env") or path.name == "credentials.json" or path.suffix in {".pem", ".key"}:
            raise ValueError(f"Potential credential file in Git selection: {name}")


def manifest_paths(root):
    """Verify the release allowlist; unrelated local files are never selected."""
    manifest = root / "MANIFEST.sha256"
    validate_git_paths(root, ["MANIFEST.sha256"])
    names = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"Invalid MANIFEST.sha256 line {number}")
        expected, name = match.groups()
        if name == "MANIFEST.sha256" or name in names:
            raise ValueError(f"Duplicate or self-referential manifest entry: {name}")
        validate_git_paths(root, [name])
        if digest(root / name) != expected:
            raise ValueError(f"Manifest checksum mismatch: {name}")
        names.append(name)
    if not names:
        raise ValueError("The publication manifest is empty")
    return names + ["MANIFEST.sha256"]


def check_git_selection(root, allowed):
    observed = command(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root).stdout.split("\0")
    unexpected = sorted(set(filter(None, observed)) - set(allowed))
    if unexpected:
        raise ValueError("Unexpected staged or nonignored files outside MANIFEST.sha256: " + ", ".join(unexpected))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_git(root, repo):
    """Only initialize a fresh directory or resume this helper's unchanged commit."""
    allowed = manifest_paths(root)
    state_path = root / ".git/rank-publication.json"
    if (root / ".git").exists() and not state_path.is_file():
        raise ValueError("Existing Git repository was not initialized by this helper; use the manual publication instructions")
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state["repo"] != repo:
            raise ValueError("Saved publication targets a different repository")
        if state.get("commit"):
            check_git_selection(root, allowed)
            head = command(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
            dirty = command(["git", "status", "--porcelain"], cwd=root).stdout.strip()
            if head != state["commit"] or dirty:
                raise ValueError("Project changed after publication began; inspect changes before resuming")
            return state_path, state
    else:
        parent = command(["git", "rev-parse", "--show-toplevel"], cwd=root, check=False)
        if parent.returncode == 0:
            raise ValueError("Project is inside another Git repository; extract it into a separate directory")
        command(["git", "init", "-b", "main"], cwd=root)
        state = {"repo": repo, "tag": TAG}
        write_json(state_path, state)
    check_git_selection(root, allowed)
    # -f admits intentionally archived inputs listed in the reviewed manifest,
    # even when general ignore rules exclude the same extension elsewhere.
    command(["git", "--literal-pathspecs", "add", "-f", "--", *allowed], cwd=root)
    command(["git", "commit", "-m", "Publish audited rank-regression reproduction and animation"], cwd=root)
    state["commit"] = command(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    write_json(state_path, state)
    return state_path, state


def verify_release_target(repo, expected_commit, release=None):
    """Resolve existing lightweight/annotated tags and the draft target."""
    reference = api(f"repos/{repo}/git/ref/tags/{quote(TAG, safe='')}")
    if reference is not None:
        obj, visited = reference["object"], set()
        while obj.get("type") == "tag":
            sha = obj["sha"]
            if sha in visited or len(visited) >= 32:
                raise ValueError("Cyclic or excessive annotated-tag chain")
            visited.add(sha)
            tag = api(f"repos/{repo}/git/tags/{sha}")
            if tag is None:
                raise ValueError("Cannot resolve annotated release tag")
            obj = tag["object"]
        if obj.get("type") != "commit" or obj.get("sha") != expected_commit:
            raise ValueError("Existing release tag points to a different commit")
    elif release is not None and not release["draft"]:
        raise ValueError("Published release tag is missing")
    if release is not None:
        if release.get("tag_name") != TAG:
            raise ValueError("Release tag name changed")
        target = release.get("target_commitish")
        if not isinstance(target, str) or not target:
            raise ValueError("Release target commit is missing")
        if target != expected_commit:
            resolved = api(f"repos/{repo}/commits/{quote(target, safe='')}")
            if resolved is None or resolved.get("sha") != expected_commit:
                raise ValueError("Release target resolves to a different commit")


def verify_remote_assets(repo, release, paths):
    by_name = {item["name"]: item for item in release["assets"]}
    for path in paths:
        entry = by_name.get(path.name)
        if not entry or entry.get("size") != path.stat().st_size:
            raise ValueError(f"Remote release asset missing or wrong size: {path.name}")
        expected = digest(path)
        remote_digest = entry.get("digest")
        if remote_digest:
            if remote_digest != "sha256:" + expected:
                raise ValueError(f"Remote release asset digest mismatch: {path.name}")
        else:
            # Older GitHub installations may not provide the asset digest.
            with tempfile.TemporaryDirectory(prefix="rank-release-verify-") as directory:
                command(["gh", "release", "download", TAG, "--repo", repo,
                         "--pattern", path.name, "--dir", directory])
                if digest(Path(directory) / path.name) != expected:
                    raise ValueError(f"Downloaded release asset digest mismatch: {path.name}")


def publish(root, repo, assets, notes):
    for name in ("git", "gh"):
        if not shutil.which(name):
            raise ValueError(f"Install {name} before using --publish")
    login = api("user")["login"]
    validate_repo(repo, login)
    existing = api("repos/" + repo)
    state_file = root / ".git/rank-publication.json"
    if existing and not state_file.is_file():
        raise ValueError("Target repository already exists; this helper will not overwrite it")
    state_path, state = prepare_git(root, repo)
    if existing:
        if not state.get("repo_created"):
            raise ValueError("Repository exists after a partial creation attempt; inspect its remote and use the manual recovery instructions")
        head = api(f"repos/{repo}/commits/main")
        if head is None or head["sha"] != state["commit"]:
            raise ValueError("Remote main no longer matches the saved publication commit; refusing to push")
    else:
        if state.get("repo_created"):
            raise ValueError("Previously created repository is no longer accessible")
        command(["gh", "repo", "create", repo, "--public", "--source", str(root),
                 "--remote", "origin", "--push", "--description",
                 "Reproducible figures and numerical experiments for high-dimensional rank regression"])
        state["repo_created"] = True
        write_json(state_path, state)
    release = find_release(repo)
    if release is None:
        verify_release_target(repo, state["commit"])
        command(["gh", "release", "create", TAG, "--repo", repo, "--draft",
                 "--target", state["commit"], "--title", "Audited reproduction " + TAG,
                 "--notes-file", str(notes)])
        release = find_release(repo)
        if release is None:
            raise RuntimeError("Draft creation returned success but the release could not be read")
        state["release_id"] = release["id"]
        write_json(state_path, state)
    elif state.get("release_id") != release["id"]:
        raise ValueError("Existing release is not recorded as this helper's draft; inspect it manually")
    verify_release_target(repo, state["commit"], release)
    if not release["draft"]:
        verify_remote_assets(repo, release, assets)
        print(f"Already published and verified: https://github.com/{repo}/releases/tag/{TAG}")
        return
    present = {item["name"] for item in release["assets"]}
    for asset in assets:
        if asset.name not in present:
            command(["gh", "release", "upload", TAG, str(asset), "--repo", repo])
    release = find_release(repo)
    if release is None or release["id"] != state["release_id"]:
        raise ValueError("Draft release changed during asset upload; inspect it manually")
    verify_release_target(repo, state["commit"], release)
    verify_remote_assets(repo, release, assets)
    command(["gh", "release", "edit", TAG, "--repo", repo, "--draft=false"])
    state["published"] = True
    write_json(state_path, state)
    print(f"Published: https://github.com/{repo}/releases/tag/{TAG}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub OWNER/NAME")
    parser.add_argument("--results", type=Path, required=True, help="Reviewed original results.tar.gz")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="Build release assets locally without contacting GitHub")
    mode.add_argument("--publish", action="store_true", help="Create a public GitHub repository and publish the verified release")
    args = parser.parse_args(argv)
    try:
        verify_input(args.results)
        print("Reviewed input archive SHA-256 verified.")
        if not (args.build or args.publish):
            print(json.dumps({"repository": args.repo, "visibility": "public", "release": TAG,
                  "git": "code, configs, summary tables, figures and GIF",
                  "release_asset": ASSET_NAME, "retained": PUBLIC_PREFIXES,
                  "omitted": "cloud launch/status records and logs", "remote_operations": 0,
                  "next": "Add --build for local assets, or --publish to publish."}, indent=2))
            return 0
        directory = ROOT / "dist/release"
        directory.mkdir(parents=True, exist_ok=True)
        asset = directory / ASSET_NAME
        record_file = directory / "asset-build.json"
        if asset.is_file() and record_file.is_file():
            record = json.loads(record_file.read_text())
            if record.get("original_archive_sha256") != INPUT_SHA256 or digest(asset) != record.get("sha256"):
                raise ValueError("Existing local release asset does not match its record; move it aside before rebuilding")
        else:
            print("Building the public raw-results archive; scientific file contents remain unchanged.", flush=True)
            record = build_asset(args.results, asset)
            write_json(record_file, record)
        checksums = directory / "SHA256SUMS.txt"
        checksums.write_text(f"{record['sha256']}  {asset.name}\n")
        notes = directory / "RELEASE_NOTES.md"
        notes.write_text(
            "Reproducibility code, corrected figure tables, and the Figure 1a scaled-risk animation.\n\n"
            "The received VM run contains 18,365 successful jobs: 18,346 retained checkpoints and 19 repairs. "
            "This is a targeted correction workflow, with documented reuse of historical results. "
            "Finite tuning grids and Monte Carlo error remain; this release does not claim an exact optimizer certificate for every historical result.\n\n"
            f"The raw-results asset retains every scientific file under `results/` and `source/` unchanged. "
            "Operational cloud logs and launch records are omitted; its manifest lists the exclusions and every retained file checksum.\n\n"
            f"Original received archive SHA-256: `{INPUT_SHA256}`.\n"
            f"Public asset SHA-256: `{record['sha256']}`.\n\n"
            "See the repository README and audit documents for figure provenance and numerical limitations. "
            "No software license has been selected on behalf of the authors.\n")
        print(json.dumps(record, indent=2))
        if args.publish:
            publish(ROOT, args.repo, [asset, checksums], notes)
        return 0
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        print(f"rank-publish: {exc}", file=sys.stderr)
        print("No rollback or deletion was attempted. A successfully created repository may already be public; an incomplete release remains a draft. See docs/publishing.md for recovery.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
