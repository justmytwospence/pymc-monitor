"""Check key pages and public API anchors after Quarto render."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

root = Path("docs/_site")
for name, expected in {"index.html": ("pymc-monitor", "Quick start"),
                       "api.html": ("API reference", "SamplingHandle"),
                       "samplers.html": ("Samplers and compatibility", "Stop lifecycle"),
                       "convergence.html": ("Convergence policies", "Checkpoints"),
                       "dashboard.html": ("Authenticated localhost", "Reusable presentation"),
                       "examples.html": ("Executable examples", "Typed event stream")}.items():
    page = root / name
    assert page.exists(), f"missing {page}"
    data = page.read_text()
    for token in expected:
        assert token in data, f"{page} missing {token}"
class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])

pages = {}
for path in root.rglob("*.html"):
    parser = Links()
    parser.feed(path.read_text())
    pages[path.resolve()] = parser
for page, parser in pages.items():
    for href in parser.links:
        url = urlsplit(href)
        if url.scheme or url.netloc:
            continue
        target = (page.parent / unquote(url.path)).resolve() if url.path else page
        if target.is_dir():
            target /= "index.html"
        assert target.exists(), f"{page}: broken link {href}"
        if url.fragment and target in pages:
            assert unquote(url.fragment) in pages[target].ids, f"{page}: broken anchor {href}"
print("documentation pages and local links OK")
