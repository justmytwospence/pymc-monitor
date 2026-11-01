#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export QUARTO_PYTHON="${QUARTO_PYTHON:-$PWD/.venv/bin/python}"
cd docs
../.venv/bin/quartodoc build
../.venv/bin/python - <<'PY'
import json
import tomllib
from pathlib import Path

path = Path("objects.json")
objects = json.loads(path.read_text())
objects["version"] = tomllib.loads(Path("../pyproject.toml").read_text())["project"]["version"]
path.write_text(json.dumps(objects) + "\n")
for page in Path("reference").glob("*.qmd"):
    page.write_text(page.read_text().rstrip() + "\n")
PY
quarto render . --to html
