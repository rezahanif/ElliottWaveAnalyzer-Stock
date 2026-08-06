import { spawn } from "node:child_process";
import path from "node:path";
import { useRuntimeConfig } from "#imports";
import { useDb } from "./db";

export interface TriggerJobInput {
  assetId: number;
  timeframe: string;
  action: string; // currently always 'run' — kept for future actions (e.g. 'train')
}

/**
 * Resolves (assetId, timeframe) against asset_timeframes — the allow-list
 * seeded by scripts/migrate_dashboard_schema.py — and only ever spawns the
 * exact script_path/job_action stored there. The API never accepts a raw
 * command from the client; this is the boundary described in
 * docs/dashboard-architecture.md §6.
 */
export function triggerJob(input: TriggerJobInput): { jobId: number } {
  const db = useDb();

  const row = db
    .prepare(
      `SELECT at.script_path, at.job_action, a.symbol
       FROM asset_timeframes at
       JOIN assets a ON a.id = at.asset_id
       WHERE at.asset_id = ? AND at.timeframe = ?`
    )
    .get(input.assetId, input.timeframe) as
    | { script_path: string; job_action: string; symbol: string }
    | undefined;

  if (!row) {
    throw createError({
      statusCode: 400,
      statusMessage: `No allow-listed script for asset_id=${input.assetId} timeframe=${input.timeframe}`,
    });
  }

  const alreadyRunning = db
    .prepare(
      `SELECT id FROM jobs WHERE asset_id = ? AND timeframe = ? AND status IN ('queued','running')`
    )
    .get(input.assetId, input.timeframe);

  if (alreadyRunning) {
    throw createError({
      statusCode: 409,
      statusMessage: "A job for this asset/timeframe is already running",
    });
  }

  const insert = db
    .prepare(
      `INSERT INTO jobs (asset_id, timeframe, action, status, started_at)
       VALUES (?, ?, ?, 'running', CURRENT_TIMESTAMP)`
    )
    .run(input.assetId, input.timeframe, input.action);

  const jobId = Number(insert.lastInsertRowid);

  const config = useRuntimeConfig();
  const repoRoot = path.resolve(process.cwd(), config.repoRoot as string);
  const args = [row.script_path, ...row.job_action.split(" ")];

  const child = spawn(config.pythonBin as string, args, {
    cwd: repoRoot,
    env: process.env,
  });

  let logTail = "";
  const appendLog = (chunk: Buffer) => {
    logTail = (logTail + chunk.toString()).slice(-4000); // keep last ~4KB
    db.prepare(`UPDATE jobs SET log_tail = ? WHERE id = ?`).run(logTail, jobId);
  };

  child.stdout.on("data", appendLog);
  child.stderr.on("data", appendLog);

  child.on("close", (code) => {
    db.prepare(
      `UPDATE jobs SET status = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?`
    ).run(code === 0 ? "done" : "failed", jobId);
  });

  return { jobId };
}
