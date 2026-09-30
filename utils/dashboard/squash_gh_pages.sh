#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Replace the publication branch's history with one commit of its current tree.
#
# Nothing on gh-pages needs history: mike reads versions.json and the kernel
# checks and dashboard read their JSON from the working tree. History only
# costs: every docs deploy adds tens of MB of regenerated Doxygen HTML, which
# every `git clone` of the repository then downloads. Each writer runs this
# after its push, while it holds the gh-pages-publish lock, so the branch
# stays at working-tree size.
#
# The tree is taken from the branch tip as fetched, so the content cannot
# change: the new commit is checked to have the same tree, and the push is
# a lease on the tip that was fetched, so anything pushed in between makes it
# fail rather than be overwritten. A branch already at one commit is left
# alone.
#
#   squash_gh_pages.sh [remote] [branch]
set -euo pipefail

remote=${1:-origin}
branch=${2:-gh-pages}

git fetch --quiet "$remote" "$branch"
old=$(git rev-parse FETCH_HEAD)
# The commit object names its parents even in a shallow fetch.
parents=$(git cat-file -p "$old" | grep -c '^parent ' || true)
if [ "$parents" -eq 0 ]; then
  echo "$remote/$branch is already a single commit ($old)"
  exit 0
fi

tree=$(git rev-parse "$old^{tree}")
new=$(git -c user.name="github-actions[bot]" \
          -c user.email="github-actions[bot]@users.noreply.github.com" \
        commit-tree "$tree" -m "$branch snapshot of $old")
if ! git diff --quiet "$old" "$new"; then
  echo "::error::the snapshot's tree differs from $old; not pushing" >&2
  exit 1
fi
git push --quiet --force-with-lease="refs/heads/$branch:$old" "$remote" "$new:refs/heads/$branch"
echo "$remote/$branch: $old -> $new (one commit, same tree)"
