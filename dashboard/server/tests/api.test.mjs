/**
 * Nitro API integration tests — boot the BUILT server (.output/server/index.mjs)
 * against a scratch DB + fixture data dir, then exercise all four read routes:
 *   /api/assets, /api/predictions, /api/candles, /api/pivots
 *
 * Run:  npm run build && node --test tests/
 * Requires the server to be built first (nitropack build).
 *
 * Fixtures live in a per-run temp dir — never touches the real data/ directory.
 */
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 3311 + Math.floor(Math.random() * 500);
const BASE = `http://127.0.0.1:${PORT}`;

let fixtureDir;
let server;
let serverLog = "";

const OHLCV_FILE = {
  asset: "BTCUSD",
  timeframe: "1D",
  columns: ["timestamp_ms", "open", "high", "low", "close", "volume"],
  data: [
    [1609459200000, 100, 110, 90, 105, 1000],
    [1609545600000, 105, 115, 95, 110, 1200],
    [1609632000000, 110, 120, 100, 115, 1100],
  ],
};

const STUB_JOB = `const [delay = '0', code = '0'] = process.argv.slice(2);\nsetTimeout(() => { console.log('stub job ran'); process.exit(Number(code)); }, Number(delay) * 1000);\n`;

const PIVOTS_FILE = {
  asset: "BTCUSD",
  timeframe: "1D",
  macro: [
    {
      timestamp_ms: 1609545600000, price: 95, swing_type: "Low", bar_index: 1,
      layer: "macro", degree: "intermediate", structure_label: "HL",
    },
  ],
  micro: [
    {
      timestamp_ms: 1609632000000, price: 120, swing_type: "High", bar_index: 2,
      layer: "micro", degree: "minor", structure_label: "UNKNOWN",
    },
  ],
};

async function waitReady(timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${BASE}/api/assets`);
      if (res.ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`server did not become ready; log:\n${serverLog}`);
}

before(async () => {
  fixtureDir = mkdtempSync(join(tmpdir(), "nitro-test-"));
  mkdirSync(join(fixtureDir, "data", "ohlcv"), { recursive: true });
  mkdirSync(join(fixtureDir, "data", "pivots"), { recursive: true });
  mkdirSync(join(fixtureDir, "scripts"), { recursive: true });
  writeFileSync(join(fixtureDir, "scripts", "stub_job.mjs"), STUB_JOB);

  // Seed DB with the shared schema + one real-looking prediction row.
  const { default: Database } = await import("better-sqlite3");
  const db = new Database(join(fixtureDir, "data", "predictions.db"));
  db.exec(`
    CREATE TABLE predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      asset TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
      timeframe TEXT, direction TEXT, wave_degree TEXT,
      btc_close_at_signal REAL, cluster_valid INTEGER,
      cluster_upper REAL, cluster_lower REAL, cluster_strength REAL,
      cluster_strength_adj REAL, target_a REAL, target_b REAL,
      scenario_a_price REAL, scenario_b_price REAL, invalidation_level REAL,
      c_top REAL, b_low REAL,
      q10_7d REAL, q50_7d REAL, q90_7d REAL,
      q10_14d REAL, q50_14d REAL, q90_14d REAL,
      q10_30d REAL, q50_30d REAL, q90_30d REAL,
      q10_60d REAL, q50_60d REAL, q90_60d REAL,
      calendar_risk_flag TEXT, macro_pivot_count INTEGER, micro_pivot_count INTEGER,
      actual_outcome TEXT, prediction_correct INTEGER,
      ob_conviction REAL, ob_bid_ask_imbalance REAL,
      ob_dominant_exchange TEXT, ob_flag TEXT
    );
    CREATE TABLE assets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT UNIQUE NOT NULL, display_name TEXT NOT NULL,
      class TEXT NOT NULL, currency TEXT, status TEXT NOT NULL DEFAULT 'planned',
      checkpoint_path TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE asset_timeframes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      asset_id INTEGER NOT NULL, timeframe TEXT NOT NULL,
      trained INTEGER NOT NULL DEFAULT 0, script_path TEXT, job_action TEXT,
      UNIQUE(asset_id, timeframe)
    );
    CREATE TABLE jobs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      asset_id INTEGER NOT NULL, timeframe TEXT NOT NULL,
      action TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
      log_tail TEXT, started_at DATETIME, finished_at DATETIME,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    INSERT INTO assets (id, symbol, display_name, class, currency, status)
      VALUES (1, 'BTC', 'Bitcoin', 'crypto', 'USD', 'active');
    INSERT INTO asset_timeframes (asset_id, timeframe, trained, script_path, job_action)
      VALUES (1, '1D', 1, 'scripts/stub_job.mjs', '["0.2", "0"]');
    INSERT INTO predictions (asset, timeframe, direction, wave_degree, btc_close_at_signal,
      cluster_upper, cluster_lower, invalidation_level, q10_7d, q50_7d, q90_7d,
      macro_pivot_count, micro_pivot_count)
      VALUES ('BTC', '1D', 'long', 'intermediate', 67000, 67000, 63000, 61000,
              62000, 65000, 68000, 143, 433);
  `);
  db.close();

  writeFileSync(
    join(fixtureDir, "data", "ohlcv", "BTC_1D.json"),
    JSON.stringify(OHLCV_FILE)
  );
  writeFileSync(
    join(fixtureDir, "data", "pivots", "BTC_1D_pivots.json"),
    JSON.stringify(PIVOTS_FILE)
  );

  server = spawn("node", [join(ROOT, ".output", "server", "index.mjs")], {
    env: {
      ...process.env,
      PORT: String(PORT),
      NITRO_DB_PATH: join(fixtureDir, "data", "predictions.db"),
      NITRO_REPO_ROOT: fixtureDir,
      NITRO_PYTHON_BIN: process.execPath,
      NITRO_JOB_TIMEOUT_MS: "5000",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  server.stdout.on("data", (d) => (serverLog += d));
  server.stderr.on("data", (d) => (serverLog += d));

  await waitReady();
});

after(() => {
  server?.kill("SIGTERM");
  rmSync(fixtureDir, { recursive: true, force: true });
});

test("GET /api/assets returns registry with nested timeframes", async () => {
  const res = await fetch(`${BASE}/api/assets`);
  assert.equal(res.status, 200);
  const body = await res.json();
  const btc = body.assets.find((a) => a.symbol === "BTC");
  assert.ok(btc, "BTC present");
  assert.deepEqual(btc.timeframes, [{ timeframe: "1D", trained: true }]);
});

test("GET /api/predictions returns rows with wave_degree", async () => {
  const res = await fetch(`${BASE}/api/predictions?asset=BTC&timeframe=1D`);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.count, 1);
  const p = body.predictions[0];
  assert.equal(p.wave_position, "long");
  assert.equal(p.wave_degree, "intermediate"); // Task 3 fix — no longer "—"
  assert.equal(p.cluster_upper, 67000);
});

test("GET /api/predictions rejects missing params with 400", async () => {
  const res = await fetch(`${BASE}/api/predictions`);
  assert.equal(res.status, 400);
});

test("GET /api/candles returns candlestick bars", async () => {
  const res = await fetch(`${BASE}/api/candles?asset=BTC&timeframe=1D`);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.count, 3);
  assert.deepEqual(body.candles[0], {
    time: 1609459200, open: 100, high: 110, low: 90, close: 105,
  });
});

test("GET /api/candles respects limit", async () => {
  const res = await fetch(`${BASE}/api/candles?asset=BTC&timeframe=1D&limit=2`);
  const body = await res.json();
  assert.equal(body.count, 2);
});

test("GET /api/pivots returns macro+micro separated by layer", async () => {
  const res = await fetch(`${BASE}/api/pivots?asset=BTC&timeframe=1D`);
  assert.equal(res.status, 200);
  const all = await res.json();
  assert.equal(all.count, 2);

  const macro = await (await fetch(`${BASE}/api/pivots?asset=BTC&timeframe=1D&layer=macro`)).json();
  assert.equal(macro.count, 1);
  assert.equal(macro.pivots[0].layer, "macro");
  assert.equal(macro.pivots[0].degree, "intermediate");

  const micro = await (await fetch(`${BASE}/api/pivots?asset=BTC&timeframe=1D&layer=micro`)).json();
  assert.equal(micro.count, 1);
  assert.equal(micro.pivots[0].layer, "micro");
});

test("GET /api/pivots rejects bad layer with 400", async () => {
  const res = await fetch(`${BASE}/api/pivots?asset=BTC&timeframe=1D&layer=bogus`);
  assert.equal(res.status, 400);
});

test("POST /api/jobs runs allow-listed argv and reaches done", async () => {
  const res = await fetch(`${BASE}/api/jobs`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ assetId: 1, timeframe: "1D" }),
  });
  assert.equal(res.status, 200);
  const { jobId } = await res.json();
  let job;
  for (let i = 0; i < 30; i++) {
    job = await (await fetch(`${BASE}/api/jobs/${jobId}`)).json();
    if (job.status !== "running") break;
    await new Promise((r) => setTimeout(r, 50));
  }
  assert.equal(job.status, "done");
  assert.match(job.log_tail, /stub job ran/);
});

test("POST /api/jobs rejects duplicate running pair", async () => {
  const first = await fetch(`${BASE}/api/jobs`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ assetId: 1, timeframe: "1D" }),
  });
  assert.equal(first.status, 200);
  const second = await fetch(`${BASE}/api/jobs`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ assetId: 1, timeframe: "1D" }),
  });
  assert.equal(second.status, 409);
});

test("POST /api/jobs rejects unsupported action", async () => {
  const res = await fetch(`${BASE}/api/jobs`, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ assetId: 1, timeframe: "1D", action: "train" }),
  });
  assert.equal(res.status, 400);
});

test("GET /api/jobs/:id returns 404 for unknown job", async () => {
  const res = await fetch(`${BASE}/api/jobs/999999`);
  assert.equal(res.status, 404);
});

test("GET /api/candles 404s for missing asset data", async () => {
  const res = await fetch(`${BASE}/api/candles?asset=NOPE&timeframe=1D`);
  assert.equal(res.status, 404);
});
