# Elliott Wave Dashboard — Architecture & System Design

Status: Draft for review
Repo analyzed: `ElliottWaveAnalyzer-Stock` (github.com/rezahanif/ElliottWaveAnalyzer-Stock)

## 1. Context

The existing repo is a Python ML pipeline (TFT models via `pytorch-forecasting`) that
produces Elliott Wave / confluence forecasts for BTC (1D/4H/1W, jointly-trained model)
and a separate, single-symbol stock model for BMRI.JK. Output currently lands in
`data/predictions.db` (SQLite) via `scripts/btc/run_daily_analysis.py`, and in Telegram
messages via `scripts/stock_orchestrator.py` (which does **not** currently persist to
SQLite). No dashboard/API/frontend exists yet — `dashboard.py` is only a README
aspiration; the only "web" artifact (`elliott-web.service`) is a bare
`python -m http.server` serving a static image folder.

This document defines the interface layer to be built on top of that pipeline,
without changing the pipeline's ML internals.

## 2. Requirements

**Functional**
- Visualize predictions per asset (candles, pivots, Fib/cluster zones, quantile
  forecast bands, invalidation levels) via tabs.
- Multi-coin pluggable: only BTC works today, but adding an asset should not require
  UI code changes.
- Trigger/re-run pipeline analysis from the UI (not just view).
- Live updates when a new prediction lands or a triggered job changes state.

**Non-functional**
- Will eventually be internet-facing → needs auth.
- Single Node/host machine — Nitro and the Python pipeline are colocated, so job
  triggering can shell out directly (no internal Python API layer needed).
- Must not corrupt or race with the existing cron/systemd-scheduled pipeline runs.

## 3. Answers this design depends on (from repo inspection)

These aren't just FYI — they directly drive the "asset registry" and "UI badge"
design in §5 and §7.

- **New timeframe within an already-trained set (BTC: 1D/4H/1W):** no retrain
  needed — `asset_timeframe` is a joint static categorical in
  `src/btc/wave_model/train.py`, already covers all three.
- **New timeframe outside that set (e.g. 15m, 1H):** needs retrain/fine-tune.
  `NaNLabelEncoder(add_nan=True)` means inference won't crash on an unseen
  timeframe, but the embedding is untrained — output would not be a real signal.
- **New coin:** needs a new model, not a retrain of BTC's. `asset_timeframe` has
  only one asset value baked in (`BTC`), and feature scalers are fit to BTC's
  price/volume magnitude. The repo's own stock model is already architected as a
  fully separate checkpoint (`src/stock/train.py`: *"No BTC weights loaded at any
  point"*) — the intended pattern is one model per asset.

## 4. High-level architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Python pipeline (unchanged)                                  │
│ scripts/btc/run_daily_analysis.py, stock_orchestrator.py     │
│ cron/systemd timers → writes data/predictions.db             │
└───────────────────────────┬───────────────────────────────────┘
                             │ shared SQLite file
┌───────────────────────────▼───────────────────────────────────┐
│ Nitro server (Node/TS) — same host                            │
│  • REST: /api/assets /api/predictions /api/jobs               │
│  • SSE:  /api/stream  (new prediction + job status push)      │
│  • Job runner: spawns fixed python commands from registry,    │
│    tracks status in `jobs` table                              │
│  • Auth middleware                                            │
└───────────────────────────┬───────────────────────────────────┘
                             │ HTTP / SSE
┌───────────────────────────▼───────────────────────────────────┐
│ Vue 3 SPA                                                      │
│  • Tab per asset (driven by `assets` table)                   │
│  • lightweight-charts panel (candles + pivots + Fib + bands)  │
│  • Prediction cards, trigger button, live job log/status      │
└─────────────────────────────────────────────────────────────┘
```

**Stack: Vue 3 + Nitro**, not Go — chosen because the frontend is already Vue, and
keeping the API layer in TS means one language and shared types across the boundary
instead of hand-syncing schemas between Go and TS. Go would win on raw resource
footprint, but that's not the bottleneck here (SQLite reads + occasional subprocess
spawn), and the Python inference process already dwarfs either option's footprint.

**Charts: `lightweight-charts`** (TradingView's open-source canvas library, ~35KB,
no iframe) instead of embedding/patching the TradingView widget — you own the data
and can draw native overlays (pivots, Fib zones, quantile bands as area series,
invalidation level as a price line) directly, which is both lighter and a better
fit for this schema than fighting someone else's DOM.

## 5. Data layer changes

Three schema changes needed before the UI can be genuinely multi-coin:

**5.1 `predictions` table** — add an `asset` column via migration (`ALTER TABLE`).
Currently implicit BTC-only in `run_daily_analysis.py`.

**5.2 `assets` table (registry, DB-backed per your choice)**
```
assets
  id              INTEGER PK
  symbol          TEXT UNIQUE       -- 'BTC', 'BMRI.JK'
  display_name    TEXT
  class           TEXT              -- 'crypto' | 'stock'
  currency        TEXT
  status          TEXT              -- 'active' | 'planned'
  checkpoint_path TEXT
  created_at      DATETIME

asset_timeframes
  asset_id        INTEGER FK -> assets.id
  timeframe       TEXT              -- '1D','4H','1W'
  trained         INTEGER           -- 1 if in the model's trained category set
  script_path     TEXT              -- exact command target for job trigger
  job_action      TEXT              -- e.g. 'daily','4h' matching existing script args
```
This table is the single source of truth for: which tabs render, which
asset+timeframe combos are legal job-trigger targets (allow-list, see §6), and the
"already trained / would need retraining" badge in the UI (directly surfaces the
§3 answers instead of you having to remember them).

**5.3 `jobs` table** — tracks UI-triggered (and optionally cron) runs:
```
jobs
  id           INTEGER PK
  asset_id     INTEGER FK
  timeframe    TEXT
  action       TEXT
  status       TEXT   -- queued|running|done|failed
  log_tail     TEXT
  started_at   DATETIME
  finished_at  DATETIME
```

**Also needed:** wire `stock_orchestrator.py` to actually write to
`predictions.db` — right now it only sends Telegram messages, so BMRI wouldn't
appear in a predictions feed as-is.

## 6. Job triggering — allow-list execution, not arbitrary exec

The UI sends `{asset_id, timeframe, action}`. Nitro validates the triple against
`assets`/`asset_timeframes`, then builds the exact subprocess command from
`script_path`/`job_action` in that row — **never** string-concatenates
client-supplied values into a shell command. Before spawning, check `jobs` for an
already-running entry on that asset+timeframe (avoid colliding with a
cron-scheduled run and racing the SQLite write). Since Nitro and the pipeline are
on the same host, this is a direct `child_process.spawn` — no internal Python API
layer required.

## 7. Real-time updates

SSE, not WebSocket — this is server→client push only (new prediction row, job
status change), SSE needs no extra client library and survives reverse
proxies/tunnels more predictably than WS. Nitro polls `predictions.db` and `jobs`
every few seconds internally and only pushes on change; no SQLite triggers needed
at this scale.

## 8. Auth

Internet-facing eventually, single user, triggers real actions (job spawn) — auth
is not optional once exposed. Two viable paths:

| | Edge auth (Tailscale / Cloudflare Tunnel + Access) | App-level auth (session/JWT) |
|---|---|---|
| Code required in Nitro | None | Login route, password hash, session/cookie handling |
| Security posture | Strong — no public listener at all | Depends entirely on your implementation being correct |
| Reachable from any browser without setup | No — needs client/tunnel | Yes |

**Recommended default: edge auth.** Only build app-level auth if you specifically
need a plain public URL with no tunnel/client anywhere.

## 9. API surface (sketch)

```
GET  /api/assets                        -> registry (drives tabs)
GET  /api/predictions?asset=&timeframe= -> candles + quantile bands + zones
POST /api/jobs                          -> {asset_id, timeframe, action} trigger
GET  /api/jobs/:id                      -> status + log_tail
GET  /api/stream                        -> SSE: prediction.new, job.updated
```

## 10. Frontend structure

- One tab per row in `assets` (status `active` → usable; `planned` → visible but
  disabled, communicates "coming soon" honestly instead of hiding it).
- Per tab: `lightweight-charts` panel (candles, pivots, Fib/cluster zone shading,
  q10/q50/q90 bands, invalidation price line) + prediction summary cards + a
  "trained timeframe" badge sourced from `asset_timeframes.trained` + trigger
  button + live job log panel fed by `/api/stream`.

## 11. Rollout phases

1. Migration: add `asset` column to `predictions`, create `assets`,
   `asset_timeframes`, `jobs` tables; backfill BTC rows with `asset='BTC'`.
2. Nitro read-only API (`/api/assets`, `/api/predictions`) + Vue shell with one
   working BTC tab and `lightweight-charts`.
3. Job trigger + `jobs` table + SSE status push.
4. Wire `stock_orchestrator.py` to persist to SQLite so BMRI populates the same way.
5. Auth (edge auth first) before any internet exposure.

## 12. Open risks / notes

- `config/stock.yaml` currently has a live Telegram bot token committed in
  plaintext — move to env/secrets before anything here touches the internet.
- Cron-scheduled runs and UI-triggered runs write to the same SQLite file — the
  `jobs`-table running-check in §6 is the guard; consider a file lock too if
  writes ever get more frequent.
