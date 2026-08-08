import { spawn } from "node:child_process";
import path from "node:path";
import { useRuntimeConfig } from "#imports";
import { useDb } from "./db";

const DEFAULT_TIMEOUT_MS = 6 * 60 * 60 * 1000;

export interface TriggerJobInput {
  assetId: number;
  timeframe: string;
  action: string;
}

type AllowRow = { script_path: string; job_action: string; symbol: string };

function parseArgv(raw: string): string[] {
  try {
    const argv = JSON.parse(raw);
    if (!Array.isArray(argv) || argv.some((x) => typeof x !== "string")) throw new Error("invalid argv");
    return argv;
  } catch {
    throw createError({ statusCode: 500, statusMessage: "Invalid JSON job_action in allow-list" });
  }
}

export function triggerJob(input: TriggerJobInput): { jobId: number } {
  const db = useDb();
  const row = db.prepare(`
    SELECT at.script_path, at.job_action, a.symbol
    FROM asset_timeframes at JOIN assets a ON a.id = at.asset_id
    WHERE at.asset_id = ? AND at.timeframe = ?
  `).get(input.assetId, input.timeframe) as AllowRow | undefined;

  if (!row) throw createError({ statusCode: 400,
    statusMessage: `No allow-listed script for asset_id=${input.assetId} timeframe=${input.timeframe}` });

  let jobId: number;
  try {
    jobId = Number(db.transaction(() => {
      const running = db.prepare(`SELECT id FROM jobs WHERE asset_id = ? AND timeframe = ? AND status IN ('queued','running')`)
        .get(input.assetId, input.timeframe);
      if (running) throw createError({ statusCode: 409,
        statusMessage: "A job for this asset/timeframe is already running" });
      const result = db.prepare(`INSERT INTO jobs (asset_id, timeframe, action, status, started_at)
        VALUES (?, ?, ?, 'running', CURRENT_TIMESTAMP)`).run(input.assetId, input.timeframe, input.action);
      return result.lastInsertRowid;
    })());
  } catch (error: any) {
    if (error?.statusCode) throw error;
    throw error;
  }

  const config = useRuntimeConfig();
  const pythonBin = String(config.pythonBin || "");
  if (!pythonBin) {
    db.prepare(`UPDATE jobs SET status='failed', log_tail=?, finished_at=CURRENT_TIMESTAMP WHERE id=?`)
      .run("NITRO_PYTHON_BIN is required", jobId);
    throw createError({ statusCode: 500, statusMessage: "NITRO_PYTHON_BIN is required" });
  }
  const repoRoot = path.resolve(process.cwd(), String(config.repoRoot));
  const child = spawn(pythonBin, [row.script_path, ...parseArgv(row.job_action)], {
    cwd: repoRoot, env: process.env,
  });
  let logTail = "";
  const appendLog = (chunk: Buffer) => {
    logTail = (logTail + chunk.toString()).slice(-4000);
    db.prepare(`UPDATE jobs SET log_tail=? WHERE id=?`).run(logTail, jobId);
  };
  child.stdout.on("data", appendLog);
  child.stderr.on("data", appendLog);

  const timeoutMs = Number(config.jobTimeoutMs || DEFAULT_TIMEOUT_MS);
  const timer = setTimeout(() => {
    appendLog(Buffer.from(`\nJob timeout after ${timeoutMs}ms\n`));
    child.kill("SIGTERM");
    setTimeout(() => { if (!child.killed) child.kill("SIGKILL"); }, 5000).unref();
  }, timeoutMs);
  child.on("error", (error) => appendLog(Buffer.from(`\nSpawn error: ${error.message}\n`)));
  child.on("close", (code, signal) => {
    clearTimeout(timer);
    const failed = code !== 0 || signal !== null;
    db.prepare(`UPDATE jobs SET status=?, finished_at=CURRENT_TIMESTAMP WHERE id=?`)
      .run(failed ? "failed" : "done", jobId);
  });
  return { jobId };
}

export function registerCronJob(assetId: number, timeframe: string): number {
  const db = useDb();
  return Number(db.transaction(() => {
    const running = db.prepare(`SELECT id FROM jobs WHERE asset_id=? AND timeframe=? AND status IN ('queued','running')`)
      .get(assetId, timeframe);
    if (running) throw new Error(`job already running: ${assetId}/${timeframe}`);
    return db.prepare(`INSERT INTO jobs (asset_id,timeframe,action,status,started_at)
      VALUES (?, ?, 'cron', 'running', CURRENT_TIMESTAMP)`).run(assetId, timeframe).lastInsertRowid;
  })());
}

export function finishCronJob(jobId: number, status: "done" | "failed", logTail = ""): void {
  useDb().prepare(`UPDATE jobs SET status=?, log_tail=?, finished_at=CURRENT_TIMESTAMP WHERE id=?`)
    .run(status, logTail.slice(-4000), jobId);
}
