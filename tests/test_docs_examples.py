"""The fenced guide examples are executable even though Quarto execution is off."""
import re
from pathlib import Path

import pytest


@pytest.mark.parametrize("page", ["index.qmd", "examples.qmd"])
def test_documented_quickstart_and_stream(page):
    text = (Path(__file__).resolve().parents[1] / "docs" / page).read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    assert blocks
    namespace = {"__name__": "__docs_example__"}
    if page == "index.qmd":
        exec(blocks[0], namespace)
        assert namespace["report"]["draws"] == 8
    else:
        exec(blocks[0], namespace)
        exec(blocks[1], namespace)
        assert namespace["result"].effective_sampler == "pymc"
