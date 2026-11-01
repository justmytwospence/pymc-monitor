# pymc-monitor

Local streaming convergence diagnostics and adaptive stopping for PyMC 5.28 and 6.

Before a PyPI release, install from Git after the repository is published:

```bash
python -m pip install 'pymc-monitor @ git+https://github.com/justmytwospence/pymc-monitor.git@main'
# For the local dashboard (optional FastAPI/uvicorn):
python -m pip install 'pymc-monitor[dashboard] @ git+https://github.com/justmytwospence/pymc-monitor.git@main'
```

For development from a local checkout: `uv sync --locked --group dev --group docs --extra dashboard`.
The version is declared in `pyproject.toml` and checked against `src/pymc_monitor/__init__.py`.

```python
import pymc as pm
import pymc_monitor as monitor

with pm.Model() as model:
    mu = pm.Normal("mu")
    y = pm.Normal("y", mu, observed=[0.1, -0.2])

idata = monitor.sample(model=model, draws=1000, tune=500, until="draft")
print(monitor.read_report(idata))
```

`sample` returns the inference data. `start` returns a `SamplingHandle` with `wait()`, `stop()`, `close()`, `done`, `snapshot()` and `dashboard_url`. For custom consumers, `iter_sample(model, sample_kwargs, ...)` yields typed `PhaseUpdate`, `SamplingProgress`, `ConvergenceUpdate`, and `RunFinished` events; its `StopIteration.value` is a `SamplingResult` containing `idata`, `report`, `requested_sampler`, and `effective_sampler`. `RunControl(stop_requested=predicate)` may be shared between a caller and either API; external polling is throttled to 0.5 seconds, and failures warn once without aborting sampling. `event_to_dict`/`event_from_dict` exchange versioned events with cloud transports. JAX samplers have phase-only progress and cannot abort early; requesting adaptive stopping with JAX raises `ValueError`.

The default sampler is nutpie when installed for a continuous model, otherwise native PyMC. Custom steps, discrete free variables and native callbacks select PyMC. Automatic nutpie import/compilation failures before sampling starts warn and retry once with native PyMC; explicit sampler selection and sampling failures never fall back. The attached report records requested and effective samplers. Set `nuts_sampler` explicitly for reproducibility.

Install the `notebook` extra for notebook widgets, `nutpie` for background nutpie sampling, or `jax` for JAX samplers. `dashboard=True` starts an authenticated local loopback dashboard (install the `dashboard` extra first); pass a shared `Dashboard()` instance to reuse it across runs. No Modal or cloud cache is required.

`until=True` uses the `final` policy; `until="draft"` uses development thresholds; a dict can override target and controller settings. Diagnostics use every element of every monitored free variable and HMC sample statistics. `check_convergence(idata, until=True)` independently assesses a completed trace. Stopped runs include a final report when at least one complete post-warmup draw can be recovered.

See the [quick start](docs/index.qmd), [sampler compatibility and stop lifecycle](docs/samplers.qmd), [convergence policies](docs/convergence.qmd), [dashboard guide](docs/dashboard.qmd), [tested examples](docs/examples.qmd), [API overview](docs/api.qmd), and [generated API reference](docs/reference/index.qmd). Quarto never executes examples during rendering; the sampler and browser suites provide acceptance coverage. Use `start()` with a context manager when you want to keep a dashboard open after sampling; `sample()` closes its resources before returning.
