# Contributing

Use Python 3.11 or newer and uv. Run `scripts/lock.sh` after changing dependencies;
this resolves under your configured cooldown then removes local-only lock metadata
without changing versions. Run `uv sync --locked --group dev --group docs`,
`uv run pytest`, `uv run ruff check src tests scripts`, `scripts/build-docs.sh`,
`uv run python -m build`, and the artifact/docs checks before proposing changes.
Install `.githooks/pre-push` using `git config core.hooksPath .githooks` if desired.
Do not import cloudposterior, Modal, or a cache package from the sampler.

After resolving with the configured cooldown via `scripts/lock.sh`, a machine with
user-level uv settings may need `uv sync --frozen --group dev --group docs` to
install the portable lock without re-resolving it. CI uses `--locked` with no
user-level settings. Do not disable release-age protections to refresh dependencies.

Run the full suite with both `pymc==5.28.*` / `arviz<1` and `pymc>=6,<7` /
`arviz>=1,<2`; `numpy<2.5` is required for the older PyTensor/Numba pair. Coverage
must reach 85% for each pair. Browser checks use `npm ci`,
`npx playwright install chromium`, and `npm run test:browser`. Verify source and
installed versions with `python scripts/check_version.py`. Install each built
wheel/sdist in a separate environment and run `scripts/smoke.py` before publishing.
The publish workflow is prepared for trusted publishing; repository, PyPI environment
and publisher configuration are coordinator tasks. Do not tag, push or publish as
part of local validation.
