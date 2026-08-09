import { requireAuth } from "../utils/auth";

export default defineEventHandler((event) => {
  const path = getRequestURL(event).pathname;
  if (path === "/api/auth/login" || path === "/api/auth/logout" || !path.startsWith("/api/")) return;
  requireAuth(event);
});
