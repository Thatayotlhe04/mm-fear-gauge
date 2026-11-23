#!/usr/bin/env bash
# Pull every data-* branch pushed by the collectors into ./data (read-only, public repo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${ROOT}/.datacache"
REMOTE="${REMOTE:-https://github.com/Thatayotlhe04/mm-fear-gauge.git}"
if [ ! -d "$CACHE/.git" ]; then
  git clone -q --no-checkout "$REMOTE" "$CACHE"
fi
git -C "$CACHE" fetch -q origin '+refs/heads/data-*:refs/remotes/origin/data-*'
mkdir -p "$ROOT/data"
for ref in $(git -C "$CACHE" for-each-ref --format='%(refname:short)' 'refs/remotes/origin/data-*'); do
  label=$(echo "$ref" | sed -E 's#origin/data-([A-Z]+)-r([0-9]+)#\1#')
  run=$(echo "$ref" | sed -E 's#origin/data-([A-Z]+)-r([0-9]+)#\2#')
  dest="$ROOT/data/r${run}"
  mkdir -p "$dest"
  git -C "$CACHE" archive "$ref" "data/${label}" | tar -x -C "$dest" --strip-components=1
  echo "synced $ref -> data/r${run}/${label}"
done
