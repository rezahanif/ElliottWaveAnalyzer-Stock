import { existsSync } from "node:fs";
import { join } from "node:path";
import { useRuntimeConfig } from "#imports";
import { ohlcvFilename, ohlcvToBars, readDataJson, type OhlcvFile } from "../utils/ohlcv";

/**
 * GET /api/candles?asset=&timeframe=&limit=
 *
 * Reads the OHLCV history the Python pipeline caches as flat JSON files at
 * data/ohlcv/{asset}_{timeframe}.json (see OHLCV_DIR in run_daily_analysis.py).
 * Returns bars in the shape lightweight-charts' CandlestickSeries expects.
 *
 * Note: stock assets (BMRI.JK) are collected to parquet, not this JSON format —
 * those return 404 with a clear message until a parquet reader ships.
 */
export default defineEventHandler((event) => {
  const query = getQuery(event);
  const asset = String(query.asset || "");
  const timeframe = String(query.timeframe || "");
  const limit = Math.min(Number(query.limit || 500), 2000);

  if (!asset || !timeframe) {
    throw createError({ statusCode: 400, statusMessage: "asset and timeframe are required" });
  }

  const filename = ohlcvFilename(asset, timeframe);
  const filePath = join(useRuntimeConfig().repoRoot, "data", "ohlcv", filename);
  if (!existsSync(filePath)) {
    throw createError({
      statusCode: 404,
      statusMessage: `no OHLCV data for ${asset} ${timeframe} (stock assets use parquet and are not yet served)`,
    });
  }

  const file = readDataJson<OhlcvFile>("ohlcv", filename);
  const bars = ohlcvToBars(file);
  const trimmed = limit > 0 ? bars.slice(-limit) : bars;

  return { asset, timeframe, count: trimmed.length, candles: trimmed };
});
