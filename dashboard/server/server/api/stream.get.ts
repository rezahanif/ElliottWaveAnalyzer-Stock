import { useDb } from "../utils/db";

const POLL_INTERVAL_MS = 4000;

/**
 * GET /api/stream (SSE)
 * Polls predictions.db and jobs on an interval and pushes only diffs.
 * SSE chosen over WebSocket per docs/dashboard-architecture.md §7 — this is
 * server->client push only, and SSE degrades more gracefully through
 * reverse proxies/tunnels than WS does.
 *
 * Events sent:
 *   event: prediction.new  data: { id, asset, timeframe, generated_at }
 *   event: job.updated     data: { id, asset_id, timeframe, status }
 */
export default defineEventHandler((event) => {
  const db = useDb();

  setHeader(event, "content-type", "text/event-stream");
  setHeader(event, "cache-control", "no-cache");
  setHeader(event, "connection", "keep-alive");

  let lastPredictionId =
    (db.prepare(`SELECT MAX(id) as maxId FROM predictions`).get() as any)?.maxId || 0;
  let lastJobState = new Map<number, string>();

  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();

      const send = (eventName: string, data: unknown) => {
        controller.enqueue(
          encoder.encode(`event: ${eventName}\ndata: ${JSON.stringify(data)}\n\n`)
        );
      };

      const poll = () => {
        const newPredictions = db
          .prepare(`SELECT id, asset, timeframe, timestamp AS generated_at FROM predictions WHERE id > ? ORDER BY id`)
          .all(lastPredictionId) as any[];

        for (const p of newPredictions) {
          send("prediction.new", p);
          lastPredictionId = p.id;
        }

        const jobs = db.prepare(`SELECT id, asset_id, timeframe, status FROM jobs`).all() as any[];
        for (const j of jobs) {
          if (lastJobState.get(j.id) !== j.status) {
            send("job.updated", j);
            lastJobState.set(j.id, j.status);
          }
        }
      };

      const interval = setInterval(poll, POLL_INTERVAL_MS);
      poll();

      event.node.req.on("close", () => clearInterval(interval));
    },
  });

  return sendStream(event, stream);
});
