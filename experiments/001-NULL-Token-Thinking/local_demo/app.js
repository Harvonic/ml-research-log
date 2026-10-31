const $ = (id) => document.getElementById(id);
const form = $("compare-form");
const input = $("problem");
const compareButton = $("compare-button");
const presetButtons = [...document.querySelectorAll("[data-digits]")];
let ready = false;
let busy = false;
let history = [];

function setControls() {
  compareButton.disabled = !ready || busy;
  presetButtons.forEach((button) => { button.disabled = !ready || busy; });
  input.readOnly = busy;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function metric(id, label, value) {
  const target = $(id);
  target.replaceChildren(document.createTextNode(`${label} `));
  const strong = document.createElement("b");
  strong.textContent = value;
  target.append(strong);
}

function showResult(name, result) {
  const badge = $(`${name}-badge`);
  badge.textContent = result.correct ? "✓ Correct" : result.terminated ? "Mismatch" : "Token limit";
  badge.className = `result-badge ${result.correct ? "correct" : "incorrect"}`;
  $(`${name}-answer`).textContent = result.answer || "∅";
  $(`${name}-note`).textContent = !result.terminated
    ? "Generation stopped at the 24-token limit without EOS."
    : !result.answer ? "The model ended without producing an answer."
    : result.correct ? "Exact match with the correct sum." : "The generated answer differs from the correct sum.";
  metric(`${name}-timing`, "Generation time", `${result.milliseconds.toFixed(1)} ms`);
  metric(`${name}-tokens`, "Answer tokens", String(result.answer_tokens));
}

function renderHistory() {
  $("history-section").hidden = history.length === 0;
  $("history-body").replaceChildren();
  for (const item of history) {
    const row = document.createElement("tr");
    const values = [item.problem, item.results.baseline.answer || "∅", item.results.null.answer || "∅", item.expected];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 1 || index === 2) {
        cell.className = item.results[index === 1 ? "baseline" : "null"].correct ? "hit" : "miss";
      }
      row.append(cell);
    });
    $("history-body").append(row);
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("The model server is unavailable.");
    const status = await response.json();
    $("connection").className = "connection ready";
    $("connection-text").textContent = status.device === "cuda" ? "GPU ready · Local" : "CPU ready · Local";
    $("device-label").textContent = status.device_name;
    for (const name of ["baseline", "null"]) {
      $(`${name}-params`).textContent = status.models[name].parameters.toLocaleString("en-US");
    }
    ready = true;
  } catch (error) {
    $("connection").className = "connection offline";
    $("connection-text").textContent = "Server offline";
    $("device-label").textContent = "Start server.py, then refresh this page.";
    showError(error.message);
  }
  setControls();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ready || busy) return;
  showError("");
  busy = true;
  setControls();
  compareButton.textContent = "Running…";
  for (const name of ["baseline", "null"]) {
    $(`${name}-card`).classList.add("busy");
    $(`${name}-card`).setAttribute("aria-busy", "true");
    $(`${name}-answer`).textContent = "Generating…";
    $(`${name}-note`).textContent = "Running the pretrained checkpoint.";
    $(`${name}-badge`).className = "result-badge pending";
    $(`${name}-badge`).textContent = "Running";
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch("/api/compare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ problem: input.value }), signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The comparison could not run.");
    input.value = data.problem;
    $("expected").replaceChildren(document.createTextNode("Correct sum  "));
    const sum = document.createElement("strong");
    sum.textContent = data.expected;
    $("expected").append(sum);
    for (const name of ["baseline", "null"]) showResult(name, data.results[name]);
    history.unshift(data);
    history = history.slice(0, 6);
    renderHistory();
  } catch (error) {
    showError(error.name === "AbortError" ? "The request took too long. Check the server terminal and try again." : error.message);
    for (const name of ["baseline", "null"]) {
      $(`${name}-answer`).textContent = "—";
      $(`${name}-note`).textContent = "Enter a valid problem and try again.";
      $(`${name}-badge`).className = "result-badge";
      $(`${name}-badge`).textContent = "Not run";
      metric(`${name}-timing`, "Generation time", "—");
      metric(`${name}-tokens`, "Answer tokens", "—");
    }
    $("expected").textContent = "Waiting for a valid comparison.";
  } finally {
    clearTimeout(timeout);
    busy = false;
    for (const name of ["baseline", "null"]) {
      $(`${name}-card`).classList.remove("busy");
      $(`${name}-card`).removeAttribute("aria-busy");
    }
    compareButton.replaceChildren(document.createTextNode("Compare "));
    const arrow = document.createElement("span");
    arrow.textContent = "↗";
    arrow.setAttribute("aria-hidden", "true");
    compareButton.append(arrow);
    setControls();
  }
});

for (const button of presetButtons) {
  button.addEventListener("click", () => {
    const digits = Number(button.dataset.digits);
    const low = 10 ** (digits - 1), span = 9 * low;
    const randomOperand = () => {
      const sample = new Uint32Array(1);
      crypto.getRandomValues(sample);
      return low + Math.floor(sample[0] / 2 ** 32 * span);
    };
    input.value = `${randomOperand()} + ${randomOperand()}`;
    showError("");
    input.focus();
  });
}
$("clear-history").addEventListener("click", () => { history = []; renderHistory(); });
initialize();
