/* Hanabi LLM Agent Viewer — frontend logic */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const COLOR_LABELS = { R: "Red", Y: "Yellow", G: "Green", W: "White", B: "Blue" };
const COLORS_ORDER = ["R", "Y", "G", "W", "B"];

let eventSource = null;
let pendingEvents = [];
let playing = true;

document.addEventListener("DOMContentLoaded", () => {

// ------------------------------------------------------------------
// Config → Start game
// ------------------------------------------------------------------
$("#start-btn").addEventListener("click", async () => {
  const errEl = $("#config-error");
  errEl.textContent = "";

  const numPlayers = parseInt($("#num-players").value, 10);

  // Start game (API key is loaded from .env on the server)
  const startRes = await fetch("/api/game/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_players: numPlayers }),
  });
  const startData = await startRes.json();
  if (startData.error) { errEl.textContent = startData.error; return; }

  // Switch to board
  $("#config-panel").classList.add("hidden");
  $("#game-board").classList.remove("hidden");

  // Stream turns
  connectStream();
});

// ------------------------------------------------------------------
// SSE stream
// ------------------------------------------------------------------
function connectStream() {
  eventSource = new EventSource("/api/game/stream");

  eventSource.addEventListener("init", (e) => {
    const data = JSON.parse(e.data);
    renderBoard(data);
  });

  eventSource.addEventListener("turn", (e) => {
    const data = JSON.parse(e.data);
    const delay = parseInt($("#speed").value, 10);
    if (delay > 0) {
      pendingEvents.push(data);
      if (pendingEvents.length === 1) drainQueue();
    } else {
      applyTurn(data);
    }
  });

  eventSource.addEventListener("gameover", (e) => {
    const data = JSON.parse(e.data);
    if (eventSource) eventSource.close();
    // Allow queue to finish, then show overlay
    const show = () => {
      if (pendingEvents.length > 0) { setTimeout(show, 200); return; }
      showGameOver(data);
    };
    show();
  });

  eventSource.onerror = () => {
    if (eventSource) eventSource.close();
  };
}

function drainQueue() {
  if (pendingEvents.length === 0) return;
  const data = pendingEvents.shift();
  applyTurn(data);
  const delay = parseInt($("#speed").value, 10);
  setTimeout(drainQueue, delay);
}

// ------------------------------------------------------------------
// Board rendering
// ------------------------------------------------------------------
function renderBoard(state) {
  // Fireworks
  const fwDiv = $("#firework-stacks");
  fwDiv.innerHTML = "";
  for (const c of COLORS_ORDER) {
    const el = document.createElement("div");
    el.className = `fw-stack fw-${c}`;
    el.id = `fw-${c}`;
    el.innerHTML = `<span class="fw-val">${state.fireworks[c]}</span><span class="fw-label">${COLOR_LABELS[c]}</span>`;
    fwDiv.appendChild(el);
  }

  // Hands
  renderHands(state);

  // Tokens
  updateTokens(state);

  // Discard
  renderDiscard(state.discard_pile || []);
}

function renderHands(state) {
  const area = $("#hands-area");
  area.innerHTML = "";
  for (let i = 0; i < state.hands.length; i++) {
    const h = state.hands[i];
    const div = document.createElement("div");
    div.className = "player-hand";
    div.id = `hand-${i}`;

    const isActive = i === state.current_player;
    div.innerHTML = `<div class="ph-name${isActive ? " active" : ""}">${h.name}</div><div class="ph-cards" id="cards-${i}"></div>`;
    area.appendChild(div);

    const cardsDiv = div.querySelector(".ph-cards");
    for (const card of h.cards) {
      cardsDiv.appendChild(makeCard(card));
    }
  }
}

function makeCard(card) {
  const el = document.createElement("div");
  const col = card.color || "unknown";
  el.className = `card c-${col}`;
  el.textContent = card.rank !== null && card.rank !== undefined ? card.rank + 1 : "?";
  if (card.known_color || card.known_rank !== null) {
    const dot = document.createElement("span");
    dot.className = "hint-dot";
    el.appendChild(dot);
  }
  return el;
}

function updateTokens(state) {
  $("#score").textContent = state.score;
  $("#lives").textContent = state.life_tokens;
  $("#hints").textContent = state.info_tokens;
  $("#deck-count").textContent = state.deck_size;
  $("#turn-num").textContent = state.turn;
}

function renderDiscard(pile) {
  const div = $("#discard-pile");
  div.innerHTML = "";
  $("#discard-count").textContent = `(${pile.length})`;
  for (const c of pile) {
    const el = document.createElement("div");
    el.className = `discard-card c-${c.color}`;
    el.textContent = c.rank + 1;
    div.appendChild(el);
  }
}

// ------------------------------------------------------------------
// Apply a turn event
// ------------------------------------------------------------------
function applyTurn(data) {
  renderBoard(data);
  addLogEntry(data);
}

function addLogEntry(data) {
  const log = $("#log");
  const entry = document.createElement("div");
  let cls = "log-entry";
  if (data.success === true) cls += " success";
  else if (data.success === false) cls += " fail";

  entry.className = cls;
  entry.innerHTML = `
    <div class="le-header">
      <span class="le-player">${data.player}</span>
      <span class="le-turn">Turn ${data.turn}</span>
    </div>
    <div class="le-action">${data.action}</div>
    <div class="le-reasoning">${escapeHtml(data.reasoning || "")}</div>
  `;
  entry.addEventListener("click", () => entry.classList.toggle("expanded"));
  log.prepend(entry);
}

// ------------------------------------------------------------------
// Game over
// ------------------------------------------------------------------
function showGameOver(data) {
  const overlay = $("#gameover-overlay");
  overlay.classList.remove("hidden");
  $("#final-score").textContent = data.score;
  $("#final-fireworks").textContent =
    "Stacks: " + COLORS_ORDER.map((c) => `${COLOR_LABELS[c]} ${data.fireworks[c]}`).join(", ");
  $("#final-turns").textContent = `Completed in ${data.turns} turns`;
}

$("#restart-btn").addEventListener("click", () => {
  $("#gameover-overlay").classList.add("hidden");
  $("#game-board").classList.add("hidden");
  $("#config-panel").classList.remove("hidden");
  $("#log").innerHTML = "";
  pendingEvents = [];
});

}); // end DOMContentLoaded

// ------------------------------------------------------------------
// Util
// ------------------------------------------------------------------
function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}
