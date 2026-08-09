// Client for the Nitro dashboard API. Types mirror the server responses —
// see dashboard/server/server/api/*.ts and scripts/migrate_dashboard_schema.py.

export interface AssetTimeframe {
  timeframe: string;
  trained: boolean;
}

export interface Asset {
  id: number;
  symbol: string;
  displayName: string;
  class: "crypto" | "stock";
  currency: string;
  status: "active" | "planned";
  checkpointPath: string | null;
  timeframes: AssetTimeframe[];
}

export interface Prediction {
  id: number;
  asset?: string;
  timeframe: string;
  /** server aliases the writer's `timestamp` column here */
  generated_at: string;
  /** server aliases the writer's `direction` column here */
  wave_position: string | null;
  wave_degree: string | null;
  current_price: number | null;
  invalidation_level: number | null;
  cluster_upper: number | null;
  cluster_lower: number | null;
  q50_7d: number | null;
  q90_7d: number | null;
  [key: string]: unknown;
}

/**
 * Signal tension — live (unconfirmed) ZigZag state for one threshold layer.
 *
 * The pivot detector (src/btc/pivots/zigzag.py) only returns *confirmed*
 * pivots today; this is the shape we want once it also exposes the running
 * extreme + locked threshold for the in-progress swing (see
 * ZigZagResult.macro_live / .micro_live in the dashboard architecture doc).
 * Backend endpoint doesn't exist yet — getSignalTension() degrades to null
 * per layer until /api/signal-tension ships.
 */
export type SwingState = "SEEKING_HIGH" | "SEEKING_LOW";
export type SignalDirection = "long" | "short";

export interface SignalTensionLayer {
  layer: "macro" | "micro";
  /** which extreme the state machine is currently tracking */
  state: SwingState;
  /** SEEKING_HIGH confirms a top (short signal); SEEKING_LOW confirms a bottom (long signal) */
  direction: SignalDirection;
  /** 0-100, how far price has moved from the running extreme toward the locked threshold */
  priceProgressPct: number;
  /** bars since the last confirmed pivot on this layer */
  barsElapsed: number;
  /** minimum bars required before a new pivot can confirm (timeframe-dependent) */
  barsRequired: number;
  barsReady: boolean;
  /** exact price that would confirm the pivot right now, given the locked threshold */
  triggerPrice: number;
  lastPivot: {
    structureLabel: string | null;
    magnitudePct: number | null;
    timestamp: string | null;
  } | null;
}

export interface SignalTensionResponse {
  macro: SignalTensionLayer | null;
  micro: SignalTensionLayer | null;
}

/** Pure so it's testable independent of whatever shape the backend ends up sending. */
export function directionFromState(state: SwingState): SignalDirection {
  return state === "SEEKING_HIGH" ? "short" : "long";
}

export interface Job {
  id: number;
  asset_id: number;
  timeframe: string;
  action: string;
  status: "queued" | "running" | "done" | "failed";
  log_tail: string | null;
  started_at: string | null;
  finished_at: string | null;
}

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.statusMessage ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export interface StreamEvents {
  onPrediction?: (p: { id: number; asset: string; timeframe: string; generated_at: string }) => void;
  onJob?: (j: { id: number; asset_id: number; timeframe: string; status: string }) => void;
}

export interface Candle {
  time: number; // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface Pivot {
  timestamp_ms: number;
  price: number;
  swing_type: "High" | "Low";
  layer: "macro" | "micro";
  degree?: string;
  structure_label?: string;
  [key: string]: unknown;
}

export const api = {
  login: (password: string) => j<{ authenticated: boolean }>("/auth/login", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  }),
  logout: () => j<{ authenticated: boolean }>("/auth/logout", { method: "POST" }),
  getAssets: () => j<{ assets: Asset[] }>("/assets"),

  getPredictions: (asset: string, timeframe: string, limit = 200) =>
    j<{ predictions: Prediction[] }>(
      `/predictions?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&limit=${limit}`
    ),

  getCandles: (asset: string, timeframe: string, limit = 500) =>
    j<{ candles: Candle[] }>(
      `/candles?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&limit=${limit}`
    ),

  getPivots: (asset: string, timeframe: string, layer: "all" | "macro" | "micro" = "all") =>
    j<{ pivots: Pivot[] }>(
      `/pivots?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&layer=${layer}`
    ),

  triggerJob: (assetId: number, timeframe: string) =>
    j<{ jobId: number; status: string }>("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assetId, timeframe }),
    }),

  getJob: (id: number) => j<Job>(`/jobs/${id}`),

  /**
   * Not backed by a server route yet — returns { macro: null, micro: null }
   * on 404 instead of throwing, so the panel can render an honest "not
   * available yet" state rather than an error banner. Safe to call now;
   * starts returning real data the moment /api/signal-tension ships.
   */
  async getSignalTension(asset: string, timeframe: string): Promise<SignalTensionResponse> {
    try {
      return await j<SignalTensionResponse>(
        `/signal-tension?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`
      );
    } catch {
      return { macro: null, micro: null };
    }
  },

  subscribe(opts: StreamEvents): () => void {
    const es = new EventSource(`${BASE}/stream`);
    if (opts.onPrediction) {
      es.addEventListener("prediction.new", (e) =>
        opts.onPrediction!(JSON.parse((e as MessageEvent).data))
      );
    }
    if (opts.onJob) {
      es.addEventListener("job.updated", (e) =>
        opts.onJob!(JSON.parse((e as MessageEvent).data))
      );
    }
    return () => es.close();
  },
};
