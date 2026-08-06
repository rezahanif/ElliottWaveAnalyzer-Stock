import Database from "better-sqlite3";
import { useRuntimeConfig } from "#imports";

let _db: Database.Database | null = null;

/**
 * Shared, lazily-opened connection to the same predictions.db the Python
 * pipeline writes to. better-sqlite3 is synchronous, which is fine here —
 * this is a low-traffic single-user dashboard, not a public API.
 */
export function useDb(): Database.Database {
  if (_db) return _db;
  const config = useRuntimeConfig();
  _db = new Database(config.dbPath, { fileMustExist: true });
  _db.pragma("journal_mode = WAL"); // allow concurrent reads while the Python side writes
  return _db;
}
