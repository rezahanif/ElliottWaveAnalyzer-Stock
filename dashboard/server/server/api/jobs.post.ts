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

  const { jobId } = triggerJob({
    assetId: body.assetId,
    timeframe: body.timeframe,
    action: body.action || "run",
  });

  return { jobId, status: "running" };
});
