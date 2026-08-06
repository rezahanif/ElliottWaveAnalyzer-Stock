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

export const api = {
  getAssets: () => j<{ assets: Asset[] }>("/assets"),

  getPredictions: (asset: string, timeframe: string, limit = 200) =>
    j<{ predictions: Prediction[] }>(
      `/predictions?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&limit=${limit}`
    ),

  triggerJob: (assetId: number, timeframe: string) =>
    j<{ jobId: number; status: string }>("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assetId, timeframe }),
    }),

  getJob: (id: number) => j<Job>(`/jobs/${id}`),

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
