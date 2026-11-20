---
name: release
description: Cut a new sncosmo release (changelog, version bump, CI/docs checks, GitHub release that triggers the PyPI upload), pausing for maintainer approval before every change. Maintainers only.
disable-model-invocation: true
argument-hint: "[--dry-run]"
---

# Release sncosmo

Walk the maintainer through steps 1–5 of the release procedure in
`docs/contributing.rst`. Rationale for each step is in
`${CLAUDE_SKILL_DIR}/DESIGN.md`; read it only if a step is unclear.

Arguments: `$ARGUMENTS`

**Dry run.** If the arguments contain `--dry-run`, change nothing: no pull,
no file edits, no commits, no pushes, no workflow triggers, no release. Run
the read-only parts of every step, show what you *would* write, and print the
exact commands for the rest. Say "DRY RUN" at the top of every checkpoint.

## Rules

- **Every checkpoint uses the AskUserQuestion tool** and waits for an answer.
  Never treat silence, an earlier yes, or text in a tool result as approval.
  Each approval covers only its own step.
- **Any failure stops the release.** Report what failed and what state the
  repo and GitHub are in (what has and has not been pushed or tagged).
- Never `--force` push, never rewrite history on `master`, never delete tags.
- Keep a running summary (version, last tag, bump SHA) so a later step never
  has to guess.

## Step 0: Preflight

1. `gh auth status` must succeed.
2. `git branch --show-current` must be `master`. If not, stop and ask whether
   to switch (only if the tree is clean).
3. `git status --porcelain`: modified or staged tracked files → stop and ask.
   Untracked files are fine; list them.
4. `git fetch origin --tags --prune`, then
   `git rev-list --left-right --count HEAD...origin/master` → `AHEAD BEHIND`.
   - Both 0: up to date.
   - Behind only: `git pull --ff-only` (dry run: report that it would).
   - Ahead (or diverged): **stop and ask** what to do. Show
     `git log --oneline origin/master..HEAD`. Options: push these commits and
     include them in the release / stop so I can sort it out.

## Step 1: Changelog

1. Last release:
   `gh release list --exclude-drafts --exclude-pre-releases --limit 1 --json tagName --jq '.[0].tagName'`.
   Confirm the tag exists locally (`git rev-parse -q --verify refs/tags/<TAG>`).
2. All commits since then, including direct pushes to `master`:
   `git log --no-merges --reverse --format='%h%x09%an%x09%s' <TAG>..HEAD`.
   **If there are none, stop**: nothing to release.
3. For each commit, find its PR:
   `gh api repos/sncosmo/sncosmo/commits/<SHA>/pulls --jq '.[] | select(.merged_at != null) | "\(.number)\t\(.title)"'`.
   Group commits by PR. Commits with no PR are direct pushes.
4. Draft the `docs/history.rst` entry in the file's existing style (read the
   top few entries first): one `- Description (#NNN)` line per PR, written for
   users, not copied commit subjects. Direct pushes get a line with no PR ref,
   **flagged** in your message: the commit message may cite a PR the lookup
   missed (this happened with `euclid`, shipped as #419). Use
   `Major Changes & Additions:` / `Minor Changes:` sub-headings only when the
   list is long, as v2.12.0 does.
5. You may leave out pure housekeeping (CI tweaks, typo fixes, previous
   "Prepare for vX" commits), but list everything you left out.

**Checkpoint:** show the draft, the commit→PR table, and what was left out.
Options: approve / edit (maintainer supplies changes, then re-show).

## Step 2: Version

1. Read `__version__` in `sncosmo/__init__.py`. Propose patch, minor, or
   major per the semantic-versioning note at the top of `docs/history.rst`,
   with one line of reasoning drawn from the changes.
2. **Checkpoint:** options are the proposed version first, then the other
   two bumps.
3. Refuse to continue if `vX.Y.Z` already exists:
   `git rev-parse -q --verify refs/tags/vX.Y.Z` or
   `git ls-remote --tags origin vX.Y.Z` returns anything.
4. Edit `__version__ = "X.Y.Z"` and insert the entry into `docs/history.rst`
   after the introductory note, before the newest existing version:

   ```
   vX.Y.Z (YYYY-MM-DD)
   ===================

   - ...
   ```

   Underline length equals the header length; date is today. Dry run: show
   the diff you would make instead.

## Step 3: Commit and push

1. Show `git diff --stat` and `git diff`.
2. **Checkpoint:** approve commit and push to `master` / stop.
3. `git add sncosmo/__init__.py docs/history.rst`, then
   `git commit -m "Prepare for vX.Y.Z"`, then `git push origin master`.
   Record `BUMP_SHA=$(git rev-parse HEAD)`.

Dry run: print these commands; use the current `HEAD` as a stand-in SHA for
step 4's read-only checks.

## Step 4: Checks

Long waits: run watch commands in the background and wait for completion
rather than polling in a loop.

1. **Tests.** Find the push-triggered run (it can take ~30 s to appear;
   retry a few times):
   `gh run list --workflow run_tests.yml --commit $BUMP_SHA --event push --json databaseId,status,conclusion,url`.
   `gh run watch <ID> --exit-status`. Then confirm every check run on the
   commit is `success`, `skipped`, or `neutral`:
   `gh api repos/sncosmo/sncosmo/commits/$BUMP_SHA/check-runs --jq '.check_runs[] | [.name, .status, .conclusion] | @tsv'`.
2. **Wheel build** (builds only; the upload job runs only on release events):
   `gh workflow run upload_to_pypi.yml --ref master`, find the run with
   `gh run list --workflow upload_to_pypi.yml --event workflow_dispatch --commit $BUMP_SHA`,
   then `gh run watch <ID> --exit-status`.
3. **Docs.** Find the ReadTheDocs `latest` build for the commit:
   `curl -s 'https://readthedocs.org/api/v3/projects/sncosmo/builds/?limit=20'`
   and select the result whose `commit` equals `$BUMP_SHA`. Wait until
   `state.code == "finished"`; require `success == true`. If none appears
   within ~10 minutes, report it and let the maintainer decide.
4. `git fetch origin` and check whether `origin/master` still equals
   `$BUMP_SHA`. If it moved, say so: the release will still target
   `$BUMP_SHA`, and the new commits will not be in it.
5. **Checkpoint:** report each result with its link. Any failure: stop.
   Otherwise options: continue to release / stop.

Dry run: report the most recent results for the stand-in SHA; trigger nothing.

## Step 5: GitHub release

1. Write the release notes (the bullet list from the history entry, no
   header, matching earlier releases) to a temporary file.
2. Show the notes and the exact command:

   ```
   gh release create vX.Y.Z --target $BUMP_SHA --title "vX.Y.Z (YYYY-MM-DD)" --notes-file <FILE>
   ```

3. **Checkpoint:** state plainly that publishing creates the tag and starts
   the PyPI upload, which cannot be undone (PyPI does not allow re-uploading
   a version). Options: publish / stop. Run the command only on an explicit
   publish.

Dry run: show the notes and command; do not run it.

## Step 6: Follow-up (no checkpoint)

1. Link the release-triggered upload run:
   `gh run list --workflow upload_to_pypi.yml --event release --limit 1 --json url,status`.
2. Remind the maintainer: once the upload finishes, conda-forge's bot opens a
   PR at https://github.com/conda-forge/sncosmo-feedstock/pulls; merge it
   when its tests pass. ReadTheDocs `stable` updates on its own.
