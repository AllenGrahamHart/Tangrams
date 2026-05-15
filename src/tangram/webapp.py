from __future__ import annotations

import random
import threading
import traceback
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from tangram.client import Speaker, make_turn_client
from tangram.config import ExperimentConfig, ModelConfig, default_results_dir, default_stimuli_dir
from tangram.experiment import (
    current_git_commit,
    current_git_dirty,
    prompt_sha256,
    stimuli_sha256,
    summarize_logs,
)
from tangram.human import HumanParticipant, HumanSessionManager, HumanTurnSubmission
from tangram.logging import Manifest, load_trial_logs, utc_now_iso, write_manifest
from tangram.participants import ClientParticipant, Participant
from tangram.prompts import PROMPT_VERSION
from tangram.runner import PairRunner


ParticipantKind = Literal["human", "llm"]


def create_human_app(
    *,
    manager: HumanSessionManager,
    config: ExperimentConfig,
    participants: dict[Speaker, Participant],
    results_dir: Path | None = None,
    stimuli_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Tangram Communication Task")
    app.state.manager = manager
    app.state.config = config
    app.state.results_dir = results_dir or default_results_dir()
    app.state.stimuli_dir = stimuli_dir or default_stimuli_dir()
    app.state.participants = participants

    @app.on_event("startup")
    def startup() -> None:
        thread = threading.Thread(target=_run_web_experiment, args=(app,), daemon=True)
        app.state.experiment_thread = thread
        thread.start()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        roles = manager.snapshot()["roles"]
        role_links = "\n".join(
            f'<a class="role-link" href="/session/{role}">{role.title()} view</a>'
            for role in roles
        )
        if not role_links:
            role_links = "<p>This run has no human participant.</p>"
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tangram Task</title>
  <style>{CSS}</style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>Tangram Communication Task</h1>
      <div class="status" id="status">Run: {manager.run_id}</div>
    </header>
    <section class="panel links">{role_links}</section>
  </main>
  <script>
    async function poll() {{
      const response = await fetch('/api/experiment');
      const state = await response.json();
      document.getElementById('status').textContent = `Run: ${{state.run_id}} | ${{state.status}}`;
    }}
    poll();
    setInterval(poll, 1000);
  </script>
</body>
</html>
"""

    @app.get("/session/{role}", response_class=HTMLResponse)
    def session_page(role: Speaker) -> str:
        if role not in manager.sessions:
            raise HTTPException(status_code=404, detail=f"No human participant for role {role}")
        return PAGE_HTML.replace("__ROLE__", role)

    @app.get("/api/experiment")
    def experiment_state() -> dict:
        return manager.snapshot()

    @app.get("/api/session/{role}/state")
    def session_state(role: Speaker) -> dict:
        try:
            return manager.get_session(role).snapshot()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/session/{role}/turn")
    def submit_turn(role: Speaker, submission: HumanTurnSubmission) -> dict:
        try:
            manager.get_session(role).submit_turn(submission)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    return app


def build_participants(
    *,
    manager: HumanSessionManager,
    director: ParticipantKind,
    matcher: ParticipantKind,
    model_config: ModelConfig | None = None,
) -> dict[Speaker, Participant]:
    resolved_model_config = model_config or ModelConfig()
    participants: dict[Speaker, Participant] = {}
    if director == "human":
        participants["director"] = HumanParticipant(
            role="director",
            session=manager.create_session("director"),
        )
    else:
        participants["director"] = ClientParticipant(
            role="director",
            client=make_turn_client(resolved_model_config),
        )

    if matcher == "human":
        participants["matcher"] = HumanParticipant(
            role="matcher",
            session=manager.create_session("matcher"),
        )
    else:
        participants["matcher"] = ClientParticipant(
            role="matcher",
            client=make_turn_client(resolved_model_config),
        )
    return participants


def _run_web_experiment(app: FastAPI) -> None:
    manager: HumanSessionManager = app.state.manager
    config: ExperimentConfig = app.state.config
    results_dir: Path = app.state.results_dir
    stimuli_dir: Path = app.state.stimuli_dir
    participants: dict[Speaker, Participant] = app.state.participants
    run_id = config.resolved_run_id()
    manager.set_status("running")
    manifest = Manifest(
        run_id=run_id,
        timestamp_start=utc_now_iso(),
        config=config.model_dump(mode="json"),
        pair_ids=[0],
        git_commit=current_git_commit(),
        git_dirty=current_git_dirty(),
        prompt_version=PROMPT_VERSION,
        prompt_sha256=prompt_sha256(),
        stimuli_sha256=stimuli_sha256(stimuli_dir, config.figures),
    )
    write_manifest(results_dir, manifest)
    try:
        runner = PairRunner(
            run_id=run_id,
            pair_id=0,
            config=config,
            participants=participants,
            stimuli_dir=stimuli_dir,
            results_dir=results_dir,
            rng=random.Random(config.seed),
        )
        runner.run_pair()
        logs = load_trial_logs(results_dir / run_id)
        logs.sort(key=lambda item: (item.pair_id, item.trial))
        manifest.trial_files = [
            str(Path(f"pair_{log.pair_id}") / f"trial_{log.trial}.json") for log in logs
        ]
        manifest.timestamp_end = utc_now_iso()
        manifest.summary = summarize_logs(logs)
        write_manifest(results_dir, manifest)
        manager.set_status("completed", summary=manifest.summary)
    except Exception as exc:  # noqa: BLE001 - surface experiment failures in the web UI
        manager.set_status("error", error=f"{exc}\n{traceback.format_exc()}")


CSS = """
:root {
  color-scheme: light;
  font-family: Arial, sans-serif;
  --border: #c9ced6;
  --ink: #20242a;
  --muted: #5f6875;
  --accent: #1f6feb;
  --surface: #f6f8fa;
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: #fff; }
.shell { max-width: 1180px; margin: 0 auto; padding: 18px; }
header { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
h1 { font-size: 24px; margin: 0; }
h2 { font-size: 16px; margin: 0 0 8px; }
.status { color: var(--muted); font-size: 14px; }
.layout { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 16px; align-items: start; }
.brief { margin-bottom: 12px; color: var(--muted); font-size: 14px; line-height: 1.4; max-width: 980px; }
.panel { border: 1px solid var(--border); border-radius: 6px; padding: 12px; background: #fff; }
.figures { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
.grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; }
.card, .slot { border: 1px solid var(--border); border-radius: 6px; padding: 6px; min-height: 104px; background: var(--surface); display: flex; flex-direction: column; align-items: center; justify-content: space-between; }
.card button, .slot button { width: 100%; border: 0; background: transparent; padding: 0; cursor: pointer; }
.card.selected, .slot.selected { outline: 3px solid var(--accent); }
.label { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
img { max-width: 100%; height: 72px; object-fit: contain; display: block; }
.chat { height: 360px; overflow: auto; display: flex; flex-direction: column; gap: 8px; }
.message { border-left: 3px solid var(--border); padding: 6px 8px; background: var(--surface); white-space: pre-wrap; }
.message.director { border-left-color: #1f6feb; }
.message.matcher { border-left-color: #238636; }
.message.system { border-left-color: #8c959f; color: var(--muted); }
.composer { display: grid; gap: 8px; margin-top: 10px; }
textarea { width: 100%; min-height: 92px; resize: vertical; font: inherit; padding: 8px; border: 1px solid var(--border); border-radius: 6px; }
select, button.primary { font: inherit; padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border); background: #fff; }
button.primary { background: var(--accent); color: white; border-color: var(--accent); cursor: pointer; }
button.primary:disabled { background: #8c959f; border-color: #8c959f; cursor: not-allowed; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.links { display: flex; gap: 12px; }
.role-link { display: inline-block; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; text-decoration: none; color: var(--ink); }
.hint { color: var(--muted); font-size: 13px; }
@media (max-width: 850px) {
  .layout { grid-template-columns: 1fr; }
  .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
"""


PAGE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tangram Task</title>
  <style>""" + CSS + """</style>
</head>
<body>
  <main class="shell">
    <header>
      <h1 id="title">Tangram Task</h1>
      <div class="status" id="runStatus">Loading</div>
    </header>
    <section class="brief" id="brief"></section>
    <div class="layout">
      <section class="panel">
        <h2 id="arrangementTitle">Figures</h2>
        <div id="arrangement" class="grid"></div>
        <h2 style="margin-top: 14px;">Private Images</h2>
        <div id="images" class="figures"></div>
      </section>
      <section class="panel">
        <h2>Dialogue</h2>
        <div id="chat" class="chat"></div>
        <div class="composer">
          <div class="hint" id="turnHint">Waiting.</div>
          <textarea id="message" placeholder="Type your message"></textarea>
          <div class="controls">
            <select id="handoff"></select>
            <span class="hint" id="selectionHint"></span>
            <button class="primary" id="send">Send</button>
          </div>
        </div>
      </section>
    </div>
  </main>
  <script>
const role = "__ROLE__";
let selectedImage = null;
let selectedPosition = null;
let lastEventCount = 0;

const handoff = document.getElementById("handoff");
if (role === "director") {
  handoff.innerHTML = '<option value="yield">Yield</option><option value="continue">Continue</option><option value="done">Done</option>';
} else {
  handoff.innerHTML = '<option value="yield">Yield</option><option value="continue">Continue</option>';
}

document.getElementById("title").textContent = `Tangram Task: ${role}`;
document.getElementById("brief").textContent = role === "director"
  ? "You and your partner have the same 12 abstract figures in different orders. Your target order is shown below. Your job is to help your partner arrange their figures to match it by communicating through the dialogue. Private image numbers are local to your screen, so do not mention image numbers in messages."
  : "You and your partner have the same 12 abstract figures in different orders. Your current arrangement is shown below. Your job is to rearrange your figures to match your partner's target order by communicating through the dialogue. Private image numbers are local to your screen, so use them only for recorded placements.";
document.getElementById("send").addEventListener("click", sendTurn);

function imgTag(item) {
  return `<img src="${item.data_url}" alt="Private image ${item.image_number}">`;
}

function renderImages(state) {
  const container = document.getElementById("images");
  container.innerHTML = "";
  for (const image of state.images) {
    const div = document.createElement("div");
    div.className = "card" + (selectedImage === image.image_number ? " selected" : "");
    div.innerHTML = `<button type="button"><div class="label">Private image ${image.image_number}</div>${imgTag(image)}</button>`;
    div.addEventListener("click", () => {
      selectedImage = image.image_number;
      renderState(state);
    });
    container.appendChild(div);
  }
}

function renderArrangement(state) {
  const title = document.getElementById("arrangementTitle");
  const container = document.getElementById("arrangement");
  const slots = role === "director" ? state.target_slots : state.arrangement_slots;
  title.textContent = role === "director" ? "Your Target Order" : "Your Current Arrangement";
  container.innerHTML = "";
  for (const slot of slots) {
    const div = document.createElement("div");
    div.className = "slot" + (selectedPosition === slot.position ? " selected" : "");
    div.innerHTML = `<button type="button"><div class="label">Position ${slot.position}</div>${imgTag(slot)}<div class="label">Image ${slot.image_number}</div></button>`;
    div.addEventListener("click", () => {
      selectedPosition = slot.position;
      renderState(state);
    });
    container.appendChild(div);
  }
}

function renderChat(state) {
  const chat = document.getElementById("chat");
  if (state.events.length === lastEventCount) return;
  lastEventCount = state.events.length;
  chat.innerHTML = "";
  for (const event of state.events) {
    const div = document.createElement("div");
    div.className = `message ${event.speaker}`;
    const speaker = event.speaker === "system" ? "System" : event.speaker;
    div.textContent = `${speaker}: ${event.text || "[no spoken text]"}`;
    chat.appendChild(div);
  }
  chat.scrollTop = chat.scrollHeight;
}

function renderState(state) {
  document.getElementById("runStatus").textContent = `Trial ${state.trial || "-"} | ${state.waiting_for_turn ? "your turn" : "waiting"}`;
  document.getElementById("turnHint").textContent = state.waiting_for_turn ? `Your turn for position ${state.current_position}.` : "Waiting for the other participant.";
  document.getElementById("message").disabled = !state.waiting_for_turn;
  document.getElementById("send").disabled = !state.waiting_for_turn;
  handoff.disabled = !state.waiting_for_turn;
  renderArrangement(state);
  renderImages(state);
  renderChat(state);
  const hint = document.getElementById("selectionHint");
  if (role === "matcher") {
    hint.textContent = `Selected image: ${selectedImage || "-"} | selected position: ${selectedPosition || "-"}`;
  } else {
    hint.textContent = "";
  }
}

async function poll() {
  const response = await fetch(`/api/session/${role}/state`);
  const state = await response.json();
  renderState(state);
}

async function sendTurn() {
  const body = {
    text: document.getElementById("message").value,
    handoff: handoff.value,
  };
  if (role === "matcher" && selectedImage && selectedPosition) {
    body.figure_image_n = selectedImage;
    body.position = selectedPosition;
  }
  const response = await fetch(`/api/session/${role}/turn`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json();
    alert(error.detail || "Could not submit turn.");
    return;
  }
  document.getElementById("message").value = "";
  selectedImage = null;
  selectedPosition = null;
  await poll();
}

poll();
setInterval(poll, 1000);
  </script>
</body>
</html>
"""
