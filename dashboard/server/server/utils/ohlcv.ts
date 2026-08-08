import { readFileSync } from "node:fs";
import { join } from "node:path";
import { useRuntimeConfig } from "#imports";

/**
 * Lazy JSON file reader for the Python pipeline's data/ directory.
 * Mirrors useDb()'s lazy-load shape — resolved against repoRoot so the
 * dashboard runs from the same host checkout the pipeline writes to.
 */
let _repoRoot: string | null = null;

function repoRoot(): string {
  if (_repoRoot) return _repoRoot;
  _repoRoot = useRuntimeConfig().repoRoot;
  return _repoRoot;
}

export interface OhlcvBar {
  time: number; // unix seconds (lightweight-charts expects seconds, file stores ms)
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface OhlcvFile {
  asset: string;
  timeframe: string;
  columns: string[];
  data: (number | string)[][];
}

/** Read and parse a pipeline JSON file from data/ relative to repoRoot. */
export function readDataJson<T>(subdir: string, filename: string): T {
  const path = join(repoRoot(), "data", subdir, filename);
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

/** Filename pattern written by scripts/btc/run_daily_analysis.py (OHLCV_DIR). */
export function ohlcvFilename(asset: string, timeframe: string): string {
  return `${asset}_${timeframe}.json`;
}

/** Convert pipeline OHLCV JSON to lightweight-charts candlestick bars. */
export function ohlcvToBars(file: OhlcvFile): OhlcvBar[] {
  return file.data.map((row) => {
    const ms = Number(row[0]);
    const [open, high, low, close] = row.slice(1, 5).map(Number);
    return { time: Math.floor(ms / 1000), open, high, low, close };
  });
}
