import { useDb } from "../utils/db";

/**
 * Live predictions columns as written by scripts/btc/run_daily_analysis.py,
 * aliased to what the web client consumes (generated_at, current_price,
 * wave_position = writer's `direction`). Explicit list — immune to schema
 * drift and skips ob_* columns that only exist after the writer's Phase 3
 * migration has run once.
 */
const PREDICTION_COLS = `
  id,
  timestamp AS generated_at,
  asset,
  timeframe,
  direction AS wave_position,
  btc_close_at_signal AS current_price,
  cluster_valid,
  cluster_strength,
  cluster_strength_adj,
  cluster_upper,
  cluster_lower,
  target_a,
  target_b,
  scenario_a_price,
  scenario_b_price,
  invalidation_level,
  c_top,
  b_low,
  q10_7d, q50_7d, q90_7d,
  q10_14d, q50_14d, q90_14d,
  q10_30d, q50_30d, q90_30d,
  q10_60d, q50_60d, q90_60d,
  calendar_risk_flag,
  macro_pivot_count,
  micro_pivot_count,
  actual_outcome,
  prediction_correct
`;

export default defineEventHandler((event) => {
  const query = getQuery(event);
  const asset = String(query.asset || "");
  const timeframe = String(query.timeframe || "");
  const limit = Math.min(Number(query.limit || 200), 1000);

  if (!asset || !timeframe) {
    throw createError({ statusCode: 400, statusMessage: "asset and timeframe are required" });
  }

  const db = useDb();
  const rows = db
    .prepare(
      `SELECT ${PREDICTION_COLS} FROM predictions
       WHERE asset = ? AND timeframe = ?
       ORDER BY id DESC
       LIMIT ?`
    )
    .all(asset, timeframe, limit);

  return { asset, timeframe, count: rows.length, predictions: rows };
});
