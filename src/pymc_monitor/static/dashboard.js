import { render } from "./renderer.js";
const config = JSON.parse(document.getElementById("dashboard-config").textContent);
const token = config.token || location.hash.slice(1);
if (location.hash) history.replaceState(null, "", location.pathname);
const headers = token ? {Authorization: `Bearer ${token}`} : {};
const $ = id => document.getElementById(id);
let stopRequested = false;
async function poll() {
  try {
    const response = await fetch(config.progress_url, {headers, cache: "no-store"});
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    const state = await response.json();
    render({...state, stop_requested: state.stop_requested || stopRequested});
    if (!state.complete) setTimeout(poll, 500);
  } catch (err) { $("status").textContent = `Dashboard unavailable: ${err.message}`; }
}
$("stop").addEventListener("click", async () => {
  $("stop").disabled = true;
  try {
    const response = await fetch(config.stop_url, {method: "POST", headers});
    if (response.status === 409) { $("status").textContent = "Run already finished or stopping"; return; }
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    stopRequested = true;
    $("status").textContent = "Stop requested";
  } catch (error) {
    $("status").textContent = `Stop failed: ${error.message}`;
    $("stop").disabled = false;
  }
});
poll();
