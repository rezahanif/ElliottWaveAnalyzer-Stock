import { triggerJob } from "../utils/jobRunner";

/**
 * POST /api/jobs  { assetId: number, timeframe: string, action?: string }
 * Only ever resolves to an allow-listed script via asset_timeframes —
 * see server/utils/jobRunner.ts and docs/dashboard-architecture.md §6.
 */
export default defineEventHandler(async (event) => {
  const body = await readBody(event);

  if (typeof body?.assetId !== "number" || typeof body?.timeframe !== "string") {
    throw createError({ statusCode: 400, statusMessage: "assetId (number) and timeframe (string) are required" });
  }
  const action = body.action ?? "run";
  if (typeof action !== "string" || !["run"].includes(action)) {
    throw createError({ statusCode: 400, statusMessage: "action must be run" });
  }

  const { jobId } = triggerJob({
    assetId: body.assetId,
    timeframe: body.timeframe,
    action,
  });

  return { jobId, status: "running" };
});
