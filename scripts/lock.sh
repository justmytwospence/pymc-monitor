#!/usr/bin/env bash
# Resolve with the user's configured cooldown, then remove user-only settings
# without changing any selected package versions.
set -euo pipefail
cd "$(dirname "$0")/.."
uv lock "$@"
trap 'rm -f .lock-selections.tmp .lock-selections-after.tmp' EXIT
.venv/bin/python - <<'PY' > .lock-selections.tmp
import tomllib
with open('uv.lock', 'rb') as f:
    lock = tomllib.load(f)
for p in lock['package']:
    print(p['name'], p.get('version', 'editable'), p.get('source', {}))
PY
uv lock --no-config "$@"
.venv/bin/python - <<'PY' > .lock-selections-after.tmp
import tomllib
with open('uv.lock', 'rb') as f:
    lock = tomllib.load(f)
for p in lock['package']:
    print(p['name'], p.get('version', 'editable'), p.get('source', {}))
PY
if ! diff -u .lock-selections.tmp .lock-selections-after.tmp; then
  rm -f .lock-selections.tmp .lock-selections-after.tmp
  echo 'sanitization changed dependency selections; inspect the diff and lock manually' >&2
  exit 1
fi
rm -f .lock-selections.tmp .lock-selections-after.tmp
if grep -q 'exclude-newer' uv.lock; then
  echo 'uv.lock still includes user-specific exclude-newer settings' >&2
  exit 1
fi
echo 'uv.lock is portable and preserves configured dependency selections.'
