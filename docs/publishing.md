# Publish the code, GIF, and numerical records

This package prepares a public repository and release; downloading it does not publish anything.
The suggested repository is `sawaya-ka/rank-regression-reproducibility`.
The GIF and its cached numerical grid belong in Git, together with code, configurations, figure tables, and the rendered figures.
The large checkpoint archive belongs in the release assets.

## One command

Install Git, GitHub CLI, and Python 3.10 or later.
Authenticate with `gh auth login`, and configure your Git commit name and email if necessary.
Extract the project into its own directory outside another Git checkout.
Keep the original downloaded `results.tar.gz`; the helper accepts only the reviewed archive with SHA-256:

```text
9186dbf80fdfe6b25b67a72f05d026015a3ab022d3ed023c0c925d2db2bc48f4
```

From the project root, run this command, replacing the results path:

```bash
python tools/publish_release.py --repo sawaya-ka/rank-regression-reproducibility --results /absolute/path/to/results.tar.gz --publish
```

This command creates a **public** GitHub repository, commits and pushes the project, builds and uploads the numerical archive, verifies its remote checksum, and publishes release `v0.3.0`.
It includes `assets/animations/fig1a_scaled_risk.gif`, which the README displays directly.
It uses your existing Git and GitHub credentials; it never contacts GCP or stops a VM.
The owner in `--repo` must match the authenticated personal GitHub account.
The helper refuses to overwrite an existing unrelated repository, force-push a branch, replace a release asset, or commit an individual file of 100 MiB or larger.
It verifies every path and SHA-256 in `MANIFEST.sha256` and stages only those reviewed files plus the manifest itself.
Unexpected staged or nonignored files, modified manifest-listed files, and symlinks stop publication.
Keep private notes and credentials outside the project directory.

For an offline preview, omit `--publish`.
To build the public archive locally without publishing, use `--build` instead.
Building the archive requires roughly another 550 MB of disk space and several minutes; no simulations are rerun.

```bash
python tools/publish_release.py --repo sawaya-ka/rank-regression-reproducibility --results /absolute/path/to/results.tar.gz
python tools/publish_release.py --repo sawaya-ka/rank-regression-reproducibility --results /absolute/path/to/results.tar.gz --build
```

## What is in the large release asset?

`dist/release/rank-results-v0.3.0.tar.gz` retains every regular file under the received archive's `results/` and `source/` directories, without changing its contents.
This includes all 18,365 checkpoints, failed parent records, selected repetitions, tuning and quadrature traces, original figure tables and metadata, figures, and repair source files.
The original run fingerprints and per-file hashes remain unchanged.
The public tar archive normalizes ownership, permissions, and timestamps and omits operational cloud logs, `launch.json`, and the top-level process status record.
These omitted records contain deployment details and are not inputs to the numerical results.
`PUBLIC_ARCHIVE_MANIFEST.json` identifies the original archive checksum, every retained file checksum, and all omitted filenames.
The public archive therefore has a different checksum from the received archive; `SHA256SUMS.txt` gives the public checksum.

This is a targeted correction run, not a rerun of every historical experiment.
The final records contain 18,346 retained successful checkpoints and 19 repaired checkpoints.
See the repository audit for retained historical LS/LAD values and ridge theory, finite tuning-grid limitations, and Monte Carlo interpretation.
The archived repair source is provenance for that run; the cleaned package also integrates the repairs for fresh runs, which necessarily have a new source fingerprint.

The project does not choose a software license on the authors' behalf.
Add the intended license before publication if you want to grant reuse rights under a named license.
Update the manuscript citation or DOI when available; do not identify the corrected figures as pixel-identical reproductions of the earlier manuscript figures.

## Interrupted publication

The repository becomes public when repository creation succeeds; the release remains a draft until all assets pass verification.
No rollback, deletion, forced push, or asset replacement is attempted on failure.
The local publication state is stored in `.git/rank-publication.json`, and local release assets remain in the ignored `dist/release/` directory.

If upload or verification is interrupted, rerun the same command from the same unchanged project.
The helper accepts its own saved repository only when the remote `main` commit still matches the recorded commit.
It also resolves any existing release tag, including annotated tags, and the draft target; both must identify the recorded publication commit.
It skips assets already present, verifies them, and uploads missing assets.
If an existing asset is incomplete or has a different checksum, the helper stops rather than replacing it automatically.
Inspect the draft in GitHub and remove only the failed asset before retrying.

If repository creation itself reports failure after partially succeeding, inspect the repository and its `main` branch before continuing.
The helper intentionally does not guess whether a same-name repository is yours from that interrupted attempt.
Use Git and GitHub CLI manually in this case, or publish under a different unused repository name from a fresh project copy.

## Manual publication from an existing checkout

If you have already initialized the project with Git, use the manual route instead of the new-repository helper:

```bash
python tools/publish_release.py --repo sawaya-ka/rank-regression-reproducibility --results /absolute/path/to/results.tar.gz --build
git status --short
python - <<'PY'
from pathlib import Path
import runpy
m = runpy.run_path("tools/publish_release.py")
paths = m["manifest_paths"](Path("."))
m["check_git_selection"](Path("."), paths)
m["command"](["git", "--literal-pathspecs", "add", "-f", "--", *paths])
PY
git diff --cached --stat
git commit -m "Publish audited rank-regression reproduction and animation"
gh repo create sawaya-ka/rank-regression-reproducibility --public --source=. --remote=origin --push
gh release create v0.3.0 --repo sawaya-ka/rank-regression-reproducibility --draft --target main --title "Audited reproduction v0.3.0" --notes-file dist/release/RELEASE_NOTES.md
gh release upload v0.3.0 dist/release/rank-results-v0.3.0.tar.gz dist/release/SHA256SUMS.txt --repo sawaya-ka/rank-regression-reproducibility
```

Inspect the uploaded assets and compare the archive checksum before publishing the draft with `gh release edit v0.3.0 --draft=false --repo sawaya-ka/rank-regression-reproducibility`.
The automated route performs this check using the GitHub asset SHA-256; if GitHub does not return a digest, it downloads the asset and hashes it locally.

Command syntax follows the official manuals for [repository creation](https://cli.github.com/manual/gh_repo_create), [release creation](https://cli.github.com/manual/gh_release_create), [asset upload](https://cli.github.com/manual/gh_release_upload), and [publishing a draft](https://cli.github.com/manual/gh_release_edit).
