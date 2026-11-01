// Transport-independent DOM renderer. All run-supplied strings use textContent.
export function render(state, root = document) {
  const get = id => root.querySelector(`#${id}`);
  const el = (tag, text) => {
    const node = document.createElement(tag);
    node.textContent = String(text ?? "unavailable");
    return node;
  };
  const row = values => {
    const tr = document.createElement("tr");
    for (const value of values) tr.append(el("td", value));
    return tr;
  };
  get("status").textContent = state.complete ? "Run complete" : state.stop_requested ? "Stop requested" : "Sampling in progress";
  get("stop").disabled = Boolean(state.complete || state.stop_requested);
  get("phases").replaceChildren(...(state.phases || []).map(p => el("li", `${p.label}: ${p.detail} (${p.status})`)));
  get("chains").replaceChildren(...Object.entries(state.sampling?.chains || {}).map(([id, c]) =>
    row([id, c.phase, `${c.draw}/${c.total}`, c.divergences])));
  const convergence = state.convergence || {};
  const verdict = convergence.verdict || {};
  get("verdict").textContent = verdict.decision ? `${verdict.decision} at ${convergence.draws} draws (${verdict.final ? "final" : "checkpoint"}); ${verdict.hint || ""}` : "Awaiting diagnostics";
  get("convergence").replaceChildren(...Object.entries(convergence.params || {}).map(([name, p]) =>
    row([name, p.rhat, p.ess_bulk, p.ess_tail])));
  get("checks").replaceChildren(...Object.entries(verdict.checks || {}).map(([name, c]) =>
    el("li", `${name}: ${c.value ?? "unavailable"}; target ${c.threshold ?? "none"} (${c.ok === true ? "pass" : c.ok === false ? "fail" : "not gating"})`)));
  get("traces").replaceChildren(...Object.entries(state.traces || {}).map(([name, chains]) => {
    const figure = document.createElement("figure");
    figure.append(el("figcaption", name));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 500 100");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Trace for ${name}`);
    const finite = chains.flat().filter(v => typeof v === "number" && Number.isFinite(v));
    const lo = Math.min(...finite), hi = Math.max(...finite);
    chains.forEach((chain, index) => {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      const points = chain.map((v, i) => typeof v === "number" && Number.isFinite(v) ? `${i * 500 / Math.max(chain.length - 1, 1)},${95 - 90 * (v - lo) / (hi - lo || 1)}` : "").filter(Boolean);
      line.setAttribute("points", points.join(" "));
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", ["#60a5fa", "#fbbf24", "#34d399", "#f87171"][index % 4]);
      svg.append(line);
    });
    figure.append(svg);
    return figure;
  }));
}
