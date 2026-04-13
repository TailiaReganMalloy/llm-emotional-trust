#!/usr/bin/env bash
# Safe convenience script to add, commit and push all local changes to the current branch
# Usage:
#   scripts/push_all_to_git.sh -m "My commit message"    # prompts for confirmation
#   scripts/push_all_to_git.sh -m "msg" -y               # auto-confirm
#   scripts/push_all_to_git.sh --dry-run                 # show actions but don't run

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

print_usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [-m MESSAGE] [-y] [--dry-run]

Options:
  -m MESSAGE   Commit message to use. If omitted, a timestamped message is used.
  -y           Auto-confirm (no interactive prompt).
  --dry-run    Show what would be run but don't execute git commands.
  -h           Show this help.

This script will:
  1) check this is a git repository
  2) show `git status --porcelain` and a short summary
  3) add all changes (git add -A)
  4) commit (if there are staged changes)
  5) push to the current branch on the 'origin' remote

Note: It does not create or modify remotes. It will not push if no commit was made.
EOF
}

COMMIT_MSG=""
AUTO_YES=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      shift
      COMMIT_MSG="$1"
      shift
      ;;
    -y)
      AUTO_YES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage
      exit 2
      ;;
  esac
done

# Ensure inside a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository." >&2
  exit 3
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
REMOTE=${REMOTE:-origin}

if [[ -z "$COMMIT_MSG" ]]; then
  COMMIT_MSG="Auto: update $(date -u +"%Y-%m-%d %H:%M:%SZ")"
fi

echo "Repository: $(git rev-parse --show-toplevel)"
echo "Branch: $BRANCH"
echo "Remote: $REMOTE"

echo
echo "Git status (porcelain):"
git status --porcelain || true

CHANGES=$(git status --porcelain)
if [[ -z "$CHANGES" ]]; then
  echo "No changes to commit. Nothing to do."
  exit 0
fi

echo
echo "Planned actions:"
echo "  git add -A"
echo "  git commit -m \"$COMMIT_MSG\" (if there are staged changes)"
echo "  git push $REMOTE $BRANCH"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run mode - no commands executed."
  exit 0
fi

if [[ $AUTO_YES -eq 0 ]]; then
  read -r -p "Proceed with the above actions? [y/N] " reply
  case "$reply" in
    [yY][eE][sS]|[yY])
      ;;
    *)
      echo "Aborted by user."
      exit 1
      ;;
  esac
fi

echo
echo "Staging all changes..."
git add -A

STAGED=$(git diff --cached --name-only || true)
if [[ -z "$STAGED" ]]; then
  echo "No staged changes after 'git add -A'. Nothing to commit."
else
  echo "Committing with message: $COMMIT_MSG"
  git commit -m "$COMMIT_MSG"
fi

echo "Pushing to $REMOTE/$BRANCH..."
git push "$REMOTE" "$BRANCH"

echo "Push complete."

exit 0
