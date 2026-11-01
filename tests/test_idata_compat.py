"""arviz 0.x / 1.x inference-data compatibility."""

import numpy as np
import pymc as pm

from pymc_monitor._idata import (
    ess_tail,
    get_group,
    group_names,
    sanitize_inference_data,
)


def _idata():
    with pm.Model():
        pm.Normal("mu", 0, 1)
        pm.Normal("y", 0, 1, observed=np.zeros(5))
        return pm.sample(draws=6, tune=6, chains=2, nuts_sampler="pymc", progressbar=False, random_seed=0)


# -- compat shims -----------------------------------------------------------

def test_group_names_clean_and_version_agnostic():
    names = group_names(_idata())
    assert "posterior" in names
    assert all("/" not in n and n for n in names)  # no slashes, no empty root


def test_ess_tail_returns_positive_number():
    val = ess_tail(np.random.default_rng(0).standard_normal((2, 200)))
    assert val > 0


def test_sanitize_makes_dict_attr_netcdf_safe(tmp_path):
    idata = _idata()
    grp = get_group(idata, "posterior")
    grp.attrs["nutpie_like"] = {"settings": {"a": 1}, "b": [1, 2, 3]}  # dict-valued attr
    sanitize_inference_data(idata)
    # The dict attr must now be a serializable scalar (JSON string).
    assert isinstance(get_group(idata, "posterior").attrs.get("nutpie_like"), str)
    path = tmp_path / "out.nc"
    idata.to_netcdf(str(path))  # must not raise
    assert path.exists()
