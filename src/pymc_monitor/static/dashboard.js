import { render } from "./renderer.js";
const token = location.hash.slice(1);
// Remove the bearer token from browser history before rendering any data.
history.replaceState(null, "", location.pathname);
const headers = {Authorization: `Bearer ${token}`};
const $ = id => document.getElementById(id);
let stopRequested = false;
async function poll() {
  try {
    const response = await fetch("/state", {headers, cache: "no-store"});
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    const state = await response.json();
    render({...state, stop_requested: state.stop_requested || stopRequested});
    if (!state.complete) setTimeout(poll, 500);
  } catch (err) { $("status").textContent = `Dashboard unavailable: ${err.message}`; }
}
$("stop").addEventListener("click", async () => {
  $("stop").disabled = true;
  try {
    const response = await fetch("/stop", {method: "POST", headers});
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    stopRequested = true;
    $("status").textContent = "Stop requested";
  } catch (error) {
    $("status").textContent = `Stop failed: ${error.message}`;
    $("stop").disabled = false;
  }
});
if (token) poll(); else $("status").textContent = "Missing dashboard token";
