"""Escalation mutates only the intended setting and exhausts each ladder."""
from copy import deepcopy

import pytest

from pymc_monitor._ladder import LADDER, fixed_point, next_rung, split_nutpie_kwargs


@pytest.mark.parametrize("cause", list(LADDER))
def test_retry_ladders(cause):
    settings = {"draws": 100, "tune": 100, "extra": {}}
    applied = {}
    messages = []
    for _ in range(4):
        result = next_rung({"cause": cause, "projected_draws": 500}, settings, "nutpie", applied)
        if result:
            messages.append(result)
    assert messages
    assert next_rung({"cause": cause}, settings, "nutpie", applied) is None
    assert settings["draws"] >= 100 and settings["tune"] >= 100


def test_no_unhelpful_or_unsupported_retry():
    settings = {"draws": 100, "tune": 100, "extra": {"target_accept": .999}}
    assert next_rung({"cause": "divergences"}, settings, "pymc", {}) is None
    assert next_rung({"cause": "slow_mixing", "projected_draws": 80}, settings, "pymc", {}) is None
    assert next_rung({"cause": "unknown"}, settings, "pymc", {}) is None
    before = deepcopy(settings)
    assert "tune" in next_rung({"cause": "ess_stalled"}, settings, "pymc", {})
    assert settings["extra"] == before["extra"]


def test_fixed_point_returns_independent_copy():
    point = {"x": 1}
    fn = fixed_point(point)
    fn(4)["x"] = 9
    assert fn(5) == {"x": 1}


def test_nutpie_argument_partition():
    class Nutpie:
        @staticmethod
        def sample(compiled_model, *, cores=None, blocking=True, **kwargs):
            pass
    good, bad = split_nutpie_kwargs(Nutpie, {"cores": 2, "target_accept": .95,
                                           "blocking": False, "typo": 3})
    assert good == {"cores": 2, "target_accept": .95}
    assert bad == {"blocking": False, "typo": 3}
