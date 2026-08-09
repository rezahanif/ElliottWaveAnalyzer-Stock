import { revokeSession, expiredCookie, sessionId } from "../../utils/auth";

export default defineEventHandler((event) => {
  revokeSession(sessionId(event));
  setHeader(event, "set-cookie", expiredCookie());
  return { authenticated: false };
});
