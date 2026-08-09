import crypto from "node:crypto";
import { useRuntimeConfig } from "#imports";
import { useDb } from "./db";

const COOKIE = "dashboard_session";
const TTL_MS = 7 * 24 * 60 * 60 * 1000;

export function ensureSessionsTable() {
  useDb().exec(`CREATE TABLE IF NOT EXISTS dashboard_sessions (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, expires_at TEXT NOT NULL
  )`);
}

export function configured() {
  return Boolean(process.env.DASHBOARD_PASSWORD_HASH || useRuntimeConfig().dashboardPasswordHash);
}

export function createSession() {
  ensureSessionsTable();
  const id = crypto.randomBytes(32).toString("hex");
  const now = Date.now();
  useDb().prepare("INSERT INTO dashboard_sessions VALUES (?, ?, ?)")
    .run(id, new Date(now).toISOString(), new Date(now + TTL_MS).toISOString());
  return id;
}

export function validSession(id: string | undefined) {
  if (!id) return false;
  ensureSessionsTable();
  return Boolean(useDb().prepare(
    "SELECT id FROM dashboard_sessions WHERE id=? AND expires_at > CURRENT_TIMESTAMP"
  ).get(id));
}

export function revokeSession(id: string | undefined) {
  if (id) useDb().prepare("DELETE FROM dashboard_sessions WHERE id=?").run(id);
}

export { COOKIE };

export function cookieHeader(id: string) {
  return `${COOKIE}=${id}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${TTL_MS / 1000}`;
}

export function expiredCookie() {
  return `${COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0`;
}

export function sessionId(event: any) {
  return getCookie(event, COOKIE);
}

export function requireAuth(event: any) {
  if (!configured()) throw createError({ statusCode: 503, statusMessage: "Dashboard auth is not configured" });
  if (!validSession(sessionId(event))) throw createError({ statusCode: 401, statusMessage: "Authentication required" });
}

export function passwordHash() {
  return String(process.env.DASHBOARD_PASSWORD_HASH || useRuntimeConfig().dashboardPasswordHash || "");
}

export { TTL_MS };
