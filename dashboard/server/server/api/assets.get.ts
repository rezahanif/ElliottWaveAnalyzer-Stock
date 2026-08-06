import { useDb } from "../utils/db";

/**
 * Returns the asset registry with nested timeframes — drives the tab list
 * and the "already trained / would need retraining" badge in the UI.
 */
export default defineEventHandler(() => {
  const db = useDb();

  const assets = db
    .prepare(
      `SELECT id, symbol, display_name as displayName, class, currency, status, checkpoint_path as checkpointPath
       FROM assets ORDER BY id`
    )
    .all() as any[];

  const timeframes = db
    .prepare(
      `SELECT asset_id as assetId, timeframe, trained FROM asset_timeframes`
    )
    .all() as any[];

  for (const asset of assets) {
    asset.timeframes = timeframes
      .filter((t) => t.assetId === asset.id)
      .map((t) => ({ timeframe: t.timeframe, trained: !!t.trained }));
  }

  return { assets };
});
