"""Verify source, installed metadata, changelog and optional release tag agree."""
import json
import os
import tomllib
from importlib.metadata import version
from pathlib import Path

import pymc_monitor

expected = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
assert expected == "0.1.0"
assert pymc_monitor.__version__ == expected
assert version("pymc-monitor") == expected
assert f"## {expected}" in Path("CHANGELOG.md").read_text()
objects = Path("docs/objects.json")
if objects.exists():
    assert json.loads(objects.read_text())["version"] == expected
if os.environ.get("GITHUB_REF_TYPE") == "tag":
    assert os.environ["GITHUB_REF_NAME"] == f"v{expected}"
print(f"pymc-monitor {expected}: source, metadata and changelog agree")
