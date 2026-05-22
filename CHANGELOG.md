# Changelog

All notable changes to this fork are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Dates are
ISO-8601 in `Europe/Berlin`.

## [R35] — 2026-05 (current)

### R35-S3 — 8th audit cycle, 41 fixes (P1+P2+P3 one-pass)

Cycle summary: the user's three "live regressions" (wake-word miss, HUD
not following Spaces, slow thinking) collapsed to two underlying bugs
(F1 `control.json` never unlinked + F3 `endpoint_silence_ms=2500`). A
third independent regression — dashboard `/api/chat` confabulating about
n8n automations — was the n8nAutomation tool not being routed.

Highlights:
- **F1** — `control.json` is now unlinked after consume and ignored if
  older than 10 minutes on cold start (`pipecat_loop.py` interrupt
  poller).
- **F2** — dashboard `/api/chat` routes through `run_reply_engine` so
  the textbox has the same tool affordance as the voice path; falls
  back to raw Ollama only if the engine path errors.
- **F3** — `endpoint_silence_ms` lowered 2500 → 1000 ms in user config
  (Pipecat itself warned the 2500 ms value collapsed STT timeout).
- **F4** — `_TOKEN_RE` in `tools/selection.py` now matches Cyrillic
  (`[a-z0-9Ѐ-ӿ]+`); RU queries now reach the keyword strategy.
- **F5** — `list_workflows()` paginates via `nextCursor` (was silently
  capped at 100 workflows).
- **F6** — 32 new tests for the `n8nAutomation` voice tool dispatcher
  (`tests/test_n8n_tool.py`).
- **F7** — README documents the n8n integration with setup, voice
  phrases, template list, security model.
- **F8** — daemon rotates `~/Library/Logs/jarvis-*.log` files on
  startup (copytruncate, 50 MB threshold, 3 generations).
- HUD: `app.on('did-change-spaces')` listener + 250 ms debounce for
  `display-metrics-changed`.
- Dashboard `/api/chat` honours the voice-pipeline standby flag.
- `n8nAutomation` description expanded: "говорит ИЛИ пишет" +
  bilingual synonyms ("show me automations", "list workflows").
- `create_from_template` respects `auto_activate=false`.
- `delete_workflow` requires `force=True` at the client level (the
  voice tool passes it through after the consent gate).
- Stale `dictation_history.json.lock` cleaned; SKILL.md files
  re-chmoded 0600.
- `_keepwarm_ollama_loop` now exits cleanly on `daemon.request_stop`.
- mcp pin loosened to `>=1.13.1,<1.14.0` (1.27 upgrade tracked on
  R36 backlog).

### R35-S2 — env-discovery fix
- `load_dotenv` now searches `~/.config/jarvis/.env` regardless of
  CWD (was: only walked up from `cwd`). User's `JARVIS_N8N_API_KEY`
  is reachable from a daemon launched by launchd.

### R35-S1 — n8n integration
- New `jarvis/integrations/n8n_client.py` (REST client with retry,
  scrubbing, pagination).
- New `jarvis/tools/builtin/n8n.py` (voice tool `n8nAutomation` with
  9 ops).
- Four shipped templates: morning_briefing, async_web_research,
  notion_to_memory, telegram_relay.
- New skill `~/.config/jarvis/skills/n8n-automation/SKILL.md`.

## [R34] — 2026-04 → 2026-05

R34 was a seven-cycle audit-and-fix marathon following the original
R31-R33 architectural upgrade. Each cycle landed P1 / P2 / P3 fixes
in one pass per the user's mandate.

- **R34-S58** — seventh audit cycle (45 findings across 22 files).
- **R34-S57** — sixth audit cycle.
- **R34-S52..S56** — third through fifth audit cycles.
- **R34-S31..S51** — feature work: Mac-control, browser skills, RU
  primary, Whisper STT fixes, HUD redesign + persistent standby,
  dialog confirmation gate, web fact-check skill, loop workers,
  latency fixes.
- **R34-S1..S30** — initial Pipecat + KAOS integration shake-out:
  wake-word, dashboard, brand identity, capability gates, seed skills,
  Settings panel, CF Tunnel deploy.

## [R33] — KAOS port

- **R33-S1** Persona system (`~/.config/jarvis/persona.md`).
- **R33-S2** Memory facts (SQLite FTS5 + decay).
- **R33-S3** Audit trail (SQLite + PII redaction).
- **R33-S4** Capability gates (env-var).
- **R33-S5/6** Dashboard backend + frontend SPA.
- **R33-S7** Performance polish.

## [R32] — Skills + Pulse decision

- **R32-S1** Skills system (L1/L2/L3 progressive disclosure).
- **R32-A/B/C** KAOS / Pulse evaluation and port decisions.

## [R31] — Pipecat integration

- **R31-S1..S6** — `pipecat_loop.py` voice pipeline; adapters for
  events.jsonl, state.json, dialog history; mac_control + fast-path;
  wake-word + echo filter; integration test.
