import { useDb } from "../utils/db";

/**
 * GET /api/predictions?asset=BTC&timeframe=1D&limit=200
 * Returns rows straight from `predictions`, newest first. The chart
 * component maps these directly to lightweight-charts series (q50 line,
 * q10-q90 band, invalidation price line) — see docs/dashboard-architecture.md §9.
 */
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
      `SELECT * FROM predictions
       WHERE asset = ? AND timeframe = ?
       ORDER BY generated_at DESC
       LIMIT ?`
    )
    .all(asset, timeframe, limit);

  return { asset, timeframe, count: rows.length, predictions: rows };
});
