"""Check that built distributions contain only the intended package and no snapshot."""
import ast
import tomllib
from pathlib import Path
from tarfile import open as open_tar
from zipfile import ZipFile

version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
files = list(Path("dist").glob(f"pymc_monitor-{version}*"))
assert len(files) == 2, f"expected wheel and sdist: {files}"
for path in files:
    if path.suffix == ".whl":
        with ZipFile(path) as artifact:
            members = artifact.namelist()
    else:
        with open_tar(path, "r:gz") as artifact:
            members = artifact.getnames()
    assert any("pymc_monitor/convergence.py" in member for member in members)
    assert any("pymc_monitor/server.py" in member for member in members)
    if path.suffix != ".whl":
        assert any(member.endswith("docs/reference/index.qmd") for member in members)
    for asset in ("dashboard.html", "dashboard.js", "dashboard.css", "renderer.js"):
        assert any(f"pymc_monitor/static/{asset}" in member for member in members)
    assert any(member.endswith("LICENSE") for member in members)
    forbidden = (".extraction-source", "cloudposterior/", ".git/", "__pycache__",
                 "docs/_site", "docs/.quarto", "docs/objects.json")
    assert not any(any(part in member for part in forbidden) for member in members)
    print(path, len(members), "files OK")

metadata = tomllib.loads(Path("pyproject.toml").read_text())["project"]
assert {"fastapi", "uvicorn"} <= {entry.split(">=")[0] for entry in
                                    metadata["optional-dependencies"]["dashboard"]}
assert not any("modal" in dep or "cache" in dep for dep in metadata["dependencies"])

for source in Path("src/pymc_monitor").rglob("*.py"):
    tree = ast.parse(source.read_text())
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        assert not any(module.split(".")[0] in {"modal", "cloudposterior", "pymc_cache"}
                       for module in modules), source
