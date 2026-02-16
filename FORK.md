# Nanobot Fork — Insight AI Customizations

> Upstream: [HKUDS/nanobot](https://github.com/HKUDS/nanobot)
> Fork: [BakariSp/nanobot](https://github.com/BakariSp/nanobot)
> Last upstream merge: `3411035` (2026-02-15)

---

## Architecture: Secretary Model

Original nanobot = single agent loop (chat → tool → loop).
Our fork splits into 3 roles:

```
User (Telegram)
    │
    ├──► 零号 (default)          ├──► Jarvis (/p)
    │    Friend-like, concise    │    Technical planner
    │    L1/L2 autonomous        │    Architecture, plans
    │    Model: Sonnet           │    Model: Opus
    │                            │
    ▼                            │
┌──────────────────────────────────┐
│   Shared State (files)           │
│   task_ledger/ · MEMORY.md       │
│   Reports/ · PROGRESS.md         │
└──────────┬───────────────────────┘
           │
┌──────────▼───────────────────────┐
│   Ralph Loop (background)        │
│   Poll task_ledger/ every 5s     │
│   Execute via `claude -p`        │
│   Auto-retry ≤2x                 │
└──────────────────────────────────┘
```

### Process Hierarchy

```
watchdog.py          # Keeps doctor alive
  └─ doctor          # Health check + hot-reload gateway
       └─ gateway    # Agent loop + channels + Ralph Loop
```

### Dual-ID Task System

- **T-xxx** = Task definition (what to do, scope, risk level)
- **W-xxx** = Worker run (one execution attempt)
- Risk: L1 (read-only) → auto / L2 (safe change) → auto / L3 (dangerous) → needs `/approve`
- Storage: `~/.nanobot/data/task_ledger/`

---

## Commands

### Chat Commands (Telegram/CLI)

| Command | Effect |
|---------|--------|
| `/help` | List all commands |
| `/p` or `/jarvis` | Switch to Jarvis (planner mode) |
| `/s` | Switch back to 零号 (secretary mode) |
| `/status` | Show active tasks |
| `/approve T-xxx` | Approve L3 task |
| `/pause T-xxx` | Pause task |
| `/cancel T-xxx` | Cancel task |
| `/new` | Archive + clear session |
| `/archive` | Archive conversation to memory |
| `/discard` | Clear session without saving |

### CLI Commands

| Command | Description |
|---------|-------------|
| `nanobot onboard` | Init config + workspace |
| `nanobot agent -m "..."` | Single message |
| `nanobot agent` | Interactive REPL |
| `nanobot gateway` | Start full server (agent + channels + Ralph Loop) |
| `nanobot doctor` | Health check + gateway with hot-reload |
| `nanobot doctor --check` | Diagnostics only |
| `nanobot status` | Show config/provider status |
| `nanobot channels status` | Show channel connectivity |
| `nanobot channels login` | WhatsApp QR login |
| `nanobot cron list/add/remove/enable/run` | Scheduled tasks |

---

## Fork Changes (vs upstream)

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `nanobot/agent/prompts.py` | ~205 | 零号 + Jarvis system prompts, follow-up nudge |
| `nanobot/doctor/__init__.py` | — | Doctor module |
| `nanobot/doctor/state.py` | ~105 | Crash state persistence, last_good_commit tracking |
| `nanobot/doctor/notify.py` | ~170 | Telegram crash alerts (out-of-process, rate-limited) |
| `nanobot/doctor/git_ops.py` | ~147 | Atomic stash→checkout→verify rollback |
| `nanobot/watchdog.py` | ~72 | Supervisor: restart doctor on crash |
| `nanobot/ralph_loop.py` | ~437 | Background task executor (poll → claude -p → report) |
| `nanobot/agent/tools/task_ledger.py` | ~462 | T-xxx/W-xxx schemas, DualLedger, risk classification |

### Modified Files

| File | Change |
|------|--------|
| `nanobot/agent/loop.py` | Secretary kernel: mode switching, command parsing, notification check, proactive follow-up, Zero/Jarvis prompt routing, active task auto-inject |
| `nanobot/cli/commands.py` | `doctor` command (hot-reload + crash loop + rollback), task ledger init in `gateway`/`agent` |
| `nanobot/channels/telegram.py` | Proxy support fix |
| `nanobot/config/schema.py` | `DoctorConfig`, `TelegramConfig.proxy`, `TelegramConfig.notify_chat_ids` |

---

## Key Behaviors

1. **Chat never blocks** — work dispatched via `dispatch_worker`, Ralph Loop executes in background
2. **Proactive follow-up** — 零号 checks silent conversations (2-45 min randomized), LLM decides whether to say something
3. **Memory consolidation** — auto-summarize old messages → `MEMORY.md` + `HISTORY.md`, dedup against `SOUL.md`/`USER.md`
4. **Doctor hot-reload** — watches `.py` file changes, restarts gateway subprocess
5. **Crash loop protection** — N crashes in window → optional auto-rollback to `last_good_commit` → Telegram alert
6. **Identity in SOUL.md** — 零号 can self-modify identity via `edit_file` at runtime
