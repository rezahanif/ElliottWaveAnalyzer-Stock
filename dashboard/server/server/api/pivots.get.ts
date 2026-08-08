import { existsSync } from "node:fs";
import { join } from "node:path";
import { useRuntimeConfig } from "#imports";
import { readDataJson } from "../utils/ohlcv";

interface PivotFile {
  asset: string;
  timeframe: string;
  macro: Record<string, unknown>[];
  micro: Record<string, unknown>[];
}

/**
 * GET /api/pivots?asset=&timeframe=&layer=macro|micro
 *
 * Reads the confirmed pivots persisted by scripts/btc/run_daily_analysis.py
 * (dump_pivots → data/pivots/{asset}_{timeframe}_pivots.json). Each pivot is
 * a PivotPoint.to_dict() with timestamp_ms, price, swing_type, layer, degree.
 */
export default defineEventHandler((event) => {
  const query = getQuery(event);
  const asset = String(query.asset || "");
  const timeframe = String(query.timeframe || "");
  const layer = String(query.layer || "all");

  if (!asset || !timeframe) {
    throw createError({ statusCode: 400, statusMessage: "asset and timeframe are required" });
  }
  if (!["all", "macro", "micro"].includes(layer)) {
    throw createError({ statusCode: 400, statusMessage: "layer must be all, macro, or micro" });
  }

  const filename = `${asset}_${timeframe}_pivots.json`;
  const filePath = join(useRuntimeConfig().repoRoot, "data", "pivots", filename);
  if (!existsSync(filePath)) {
    throw createError({
      statusCode: 404,
      statusMessage: `no pivots file for ${asset} ${timeframe} — writer hasn't dumped it yet`,
    });
  }

  const file = readDataJson<PivotFile>("pivots", filename);
  const pivots =
    layer === "all"
      ? [...file.macro, ...file.micro]
      : file[layer as "macro" | "micro"];

  return {
    asset,
    timeframe,
    layer,
    count: pivots.length,
    pivots,
  };
});
