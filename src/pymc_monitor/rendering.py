"""Safe presentation primitives shared by notebook and terminal displays."""
from __future__ import annotations

import html
import math


def number(value, digits=3):
    """Render unavailable/nonfinite diagnostics without implying convergence."""
    if value is None:
        return "unavailable"
    try:
        return f"{value:.{digits}f}" if math.isfinite(value) else "unavailable"
    except (TypeError, ValueError):
        return str(value)


def convergence_summary(update):
    verdict = update.verdict or {}
    decision = verdict.get("decision", "pending")
    suffix = "final" if update.final else "checkpoint"
    summary = f"{decision} at {update.draws} draws/chain ({suffix})"
    if verdict.get("hint"):
        summary += ": " + str(verdict["hint"])
    return summary


def convergence_html(update):
    """Escape all variable names, verdict fields and values at the HTML boundary."""
    def escape(value):
        return html.escape(str(value), quote=True)
    rows = []
    for name, param in update.params.items():
        cells = [name, number(param.rhat), number(param.ess_bulk, 0), number(param.ess_tail, 0)]
        rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>")
    checks = []
    for name, check in (update.verdict or {}).get("checks", {}).items():
        outcome = {True: "pass", False: "fail", None: "not gating"}[check.get("ok")]
        checks.append(f"<li>{escape(name)}: {escape(number(check.get('value')))}; "
                      f"target {escape(number(check.get('threshold')))} ({outcome})</li>")
    return ("<section><h3>Convergence</h3><p>" + escape(convergence_summary(update)) +
            "</p><table><thead><tr><th>Variable</th><th>R-hat</th><th>Bulk ESS</th>"
            "<th>Tail ESS</th></tr></thead><tbody>" + "".join(rows) +
            "</tbody></table><ul>" + "".join(checks) + "</ul></section>")
