"""Compatibility helpers exercised against both supported ArviZ containers."""
import json
from types import SimpleNamespace

import arviz as az
import numpy as np
import pytest
import xarray as xr

from pymc_monitor import idata as compat


def data(n=10):
    groups = {"posterior": {"x": np.arange(2*n, dtype=float).reshape(2, n)},
              "sample_stats": {"energy": np.ones((2, n))},
              "observed_data": {"y": np.ones(3)}}
    try:
        return az.from_dict(**groups)
    except TypeError:
        return az.from_dict(groups)


def test_access_and_add_group():
    obj = data()
    assert compat.get_group(obj, "missing") is None
    assert compat.get_group({"posterior": "group"}, "posterior") == "group"
    group = xr.Dataset({"test": (["chain", "draw"], np.ones((2, 10)))})
    compat.add_group(obj, "log_likelihood", group)
    assert "log_likelihood" in compat.group_names(obj)
    assert compat.group_attrs(obj, "posterior") is not None
    compat.load_all(obj)
    assert compat.to_inference_data(obj) is obj
    assert compat.pymc_major() in (5, 6)


def test_trim_and_concat_all_groups():
    first = data(10)
    second = data(7)
    first.posterior["x"].values[0, 8:] = np.nan
    trimmed = compat.trim_incomplete_draws(first)
    assert trimmed.posterior.sizes["draw"] == 8
    merged = compat.concat_chains([trimmed, second])
    assert merged.posterior.sizes == {"chain": 4, "draw": 7}
    assert merged.observed_data.sizes["y_dim_0"] == 3
    assert list(merged.posterior.chain.values) == [0, 1, 2, 3]
    assert compat.trim_incomplete_draws({}) == {}


def test_sanitize_copies_nested_attrs_and_object_variables():
    obj = data()
    post = compat.get_group(obj, "posterior")
    post.attrs["mapping"] = {"a": 1}
    post.x.attrs["nested"] = {"b": 2}
    post.chain.attrs["coord"] = {"c": 3}
    post["numeric_object"] = xr.DataArray(np.ones((2, 10), dtype=object), dims=("chain", "draw"))
    sanitized = compat.sanitize_inference_data(obj, copy=True)
    result = compat.get_group(sanitized, "posterior")
    assert isinstance(post.attrs["mapping"], dict)
    assert json.loads(result.attrs["mapping"]) == {"a": 1}
    assert json.loads(result.x.attrs["nested"]) == {"b": 2}
    assert json.loads(result.chain.attrs["coord"]) == {"c": 3}
    assert result.numeric_object.dtype == float


def test_fallback_add_group_never_silently_loses_it():
    class Bare:
        groups = []
    with pytest.raises(RuntimeError, match="attach group"):
        compat.add_group(Bare(), "posterior", xr.Dataset())
    class Good:
        groups = ["posterior"]
    good = Good()
    compat.add_group(good, "posterior", xr.Dataset())
    assert good.posterior is not None


def test_load_failure_is_logged(caplog):
    def fail():
        raise RuntimeError("closed file")
    obj = SimpleNamespace(groups=["/posterior"], posterior=SimpleNamespace(load=fail))
    with caplog.at_level("DEBUG"):
        compat.load_all(obj)
    assert "could not eagerly load" in caplog.text


def test_tail_api_fallback_is_explicit(monkeypatch):
    calls = []
    def ess(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("method") == "tail":
            raise ValueError("old interface")
        return 12
    monkeypatch.setattr(az, "ess", ess)
    monkeypatch.setattr(compat, "_ess_tail_fallback_warned", False)
    with pytest.warns(UserWarning, match="bulk ESS"):
        assert compat.ess_tail(np.ones((2, 10))) == 12
    assert len(calls) == 4


def test_tail_explicit_probability_fallback(monkeypatch):
    def ess(*args, **kwargs):
        if "prob" not in kwargs:
            raise TypeError("requires prob")
        return 22
    monkeypatch.setattr(az, "ess", ess)
    assert compat.ess_tail(np.ones((2, 10))) == 22
