# `/release` skill: design

Why `SKILL.md` in this folder works the way it does. Not packaged (wheels ship
only `sncosmo/`; the sdist follows `MANIFEST.in`).

## Intent

Releasing sncosmo by hand is tedious. This skill automates the release
procedure in `docs/contributing.rst`, up to (not including) the conda-forge
merge, as a Claude Code command that pauses for maintainer approval at each
step.

**Goals**
- One command (`/release`): sync, changelog, version bump, push, CI/docs
  checks, GitHub release.
- Every change to the repo or GitHub is shown and approved before it happens.
- The changelog covers *all* commits since the last release, including direct
  pushes to `master`, not only merged PRs.

**Non-goals:** the conda-forge feedstock PR (step 6; the skill only reminds),
PyPI trusted publishing, a helper script usable without Claude.

**Anti-goals:** publishing a GitHub release (which triggers the PyPI upload)
without an explicit yes; tagging a commit that has not passed CI.

## Approaches considered

1. **Single Markdown skill committed to the repo** (chosen). Little to
   maintain, shared with co-maintainers.
2. Skill plus a Python helper for the mechanical steps. Rejected: extra code
   for a handful of one-line `git`/`gh` commands.
3. Claude Code Workflow script. Rejected: workflows run in the background and
   cannot pause for approval mid-run.

## Steps and checkpoints

| # | Step | Checkpoint |
|---|------|-----------|
| 0 | **Preflight.** On `master`, `gh` authenticated, working tree clean apart from untracked files. `git fetch origin --tags`. Behind `origin/master`: `git pull --ff-only`. Ahead or diverged: stop and ask what to do. | Only if ahead/diverged or tree dirty |
| 1 | **Changelog.** Latest release tag from `gh release list`. Read `git log <tag>..HEAD` (every commit, so direct pushes are included); attach PR numbers where a commit came from a PR. Draft a `docs/history.rst` entry in the existing style (`- Description (#NNN)`, no PR ref for direct pushes). Housekeeping commits (CI tweaks, typo fixes) may be left out, but the draft lists what was left out. | Approve or edit the draft |
| 2 | **Version.** Propose patch/minor/major with reasoning from the changes. Edit `__version__` in `sncosmo/__init__.py` and write the history header `vX.Y.Z (YYYY-MM-DD)`. | Approve the version |
| 3 | **Commit, push, start wheel build.** Show the diff; commit `Prepare for vX.Y.Z`; push directly to `master` (maintainer decision: no PR); dispatch the build-only wheel workflow once `master` is confirmed to be the bump SHA. | Approve commit, push, and wheel build |
| 4 | **Checks.** The push triggers the Tests workflow; watch the run for that exact SHA (`gh run watch`). Wait for every check run on the SHA to complete, watch the wheel build started in step 3, and confirm the ReadTheDocs `latest` build for that commit succeeded. Any failure stops the release. | Approve to continue |
| 5 | **Release.** Show the exact `gh release create vX.Y.Z --target <SHA> --title "vX.Y.Z (YYYY-MM-DD)" --notes <entry>` with notes taken from the history entry. Publishing triggers the PyPI upload. | Explicit yes required |
| 6 | **Follow-up.** Link the PyPI upload run, then remind about merging the conda-forge feedstock PR when the bot opens it. | none |

## Decisions

Maintainer:
- Changelog source is the full git log, not just PRs.
- Behind origin: pull automatically. Ahead or diverged: ask.
- The version bump is pushed directly to `master`; CI runs on that push before
  tagging.

Builder's calls (open to change):
- Running the wheel build in step 4 catches packaging failures before the
  tag exists rather than after.
- Do not allowlist `gh release create` (or blanket `gh release *`) in Claude
  Code permissions. The permission prompt is a second safeguard behind the
  skill's own explicit approval at step 5, which must never be skipped.
- The `--target <SHA>` flag pins the tag to the commit that passed CI, even if
  someone pushes to `master` during the release.
