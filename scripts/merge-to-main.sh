#!/bin/bash
set -euo pipefail

# merge-to-main.sh — Safe dev→main merge for flaircast
#
# Usage:
#   ./scripts/merge-to-main.sh           # merge + verify (no push)
#   ./scripts/merge-to-main.sh --push    # merge + verify + push
#
# What it does:
#   1. Checks out main and pulls latest
#   2. Merges only the allowed files from dev (cherry-pick --no-commit)
#   3. Removes any forbidden files that leaked in
#   4. Runs tests to verify
#   5. Commits and optionally pushes
#
# Allowed on main:
#   flaircast/, tests/, .github/, scripts/,
#   pyproject.toml, README.md, README_ja.md, LICENSE, assets/

FORBIDDEN_RE='^(research/|results/|docs/|scripts/|CLAUDE\.md)'

red()   { echo -e "\033[1;31m$*\033[0m"; }
green() { echo -e "\033[1;32m$*\033[0m"; }
bold()  { echo -e "\033[1m$*\033[0m"; }

# ── Pre-flight checks ──────────────────────────────────────────────────
if [[ $(git status --porcelain --untracked-files=no | wc -l) -gt 0 ]]; then
    red "ERROR: Working tree is dirty. Commit or stash changes first."
    exit 1
fi

current_branch=$(git branch --show-current)

# ── Step 1: Update main ───────────────────────────────────────────────
bold "Step 1: Checking out main and pulling latest..."
git checkout main
git pull origin main --ff-only || {
    red "ERROR: Cannot fast-forward main. Resolve manually."
    git checkout -f "$current_branch"
    exit 1
}

# ── Step 2: Merge dev ─────────────────────────────────────────────────
bold "Step 2: Merging dev into main..."
git merge dev --no-commit --no-ff || {
    red "Merge conflict detected. Resolving conflicts (taking dev version)..."
    conflicted=$(git diff --name-only --diff-filter=U || true)
    if [[ -n "$conflicted" ]]; then
        echo "$conflicted" | while IFS= read -r f; do
            git checkout dev -- "$f" 2>/dev/null || true
            git add "$f" 2>/dev/null || true
        done
    fi
}

# ── Step 3: Remove forbidden files ────────────────────────────────────
bold "Step 3: Removing forbidden files..."
forbidden=$(git diff --cached --name-only | grep -E "$FORBIDDEN_RE" || true)
if [[ -z "$forbidden" ]]; then
    # Also check the full tree
    forbidden=$(git ls-files | grep -E "$FORBIDDEN_RE" || true)
fi

if [[ -n "$forbidden" ]]; then
    echo "$forbidden" | xargs git rm --cached -f -- 2>/dev/null || true
    echo "$forbidden" | wc -l | xargs -I{} echo "  Removed {} forbidden files"
fi

# ── Step 4: Final verification ────────────────────────────────────────
bold "Step 4: Verifying main tree is clean..."
leaked=$(git diff --cached --name-only | grep -E "$FORBIDDEN_RE" || true)
leaked2=$(git ls-files | grep -E "$FORBIDDEN_RE" || true)
all_leaked="$leaked$leaked2"

if [[ -n "$all_leaked" ]]; then
    red "ERROR: Forbidden files still present after cleanup:"
    echo "$all_leaked"
    git merge --abort 2>/dev/null || git reset --merge
    git checkout -f "$current_branch"
    exit 1
fi

green "  Tree is clean."

# ── Step 5: Run tests ─────────────────────────────────────────────────
bold "Step 5: Installing deps and running tests..."
uv sync --extra dev --quiet
if uv run pytest tests/ --tb=short -q; then
    green "  All tests passed."
else
    red "ERROR: Tests failed. Aborting merge."
    git merge --abort 2>/dev/null || git reset --merge
    git checkout -f "$current_branch"
    exit 1
fi

# ── Step 6: Commit ────────────────────────────────────────────────────
bold "Step 6: Committing merge..."
git commit -m "feat: merge dev into main (package updates only)"

# ── Step 7: Optional push ────────────────────────────────────────────
if [[ "${1:-}" == "--push" ]]; then
    bold "Step 7: Pushing to origin/main..."
    git push origin main
    green "Pushed to origin/main."
else
    bold "Step 7: Skipped push. Run 'git push origin main' when ready."
fi

# ── Return to original branch ─────────────────────────────────────────
git checkout -f "$current_branch"
green "Done. Back on $current_branch."
