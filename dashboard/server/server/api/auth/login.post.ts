import bcrypt from "bcryptjs";
import { createSession, cookieHeader, passwordHash, configured } from "../../utils/auth";

export default defineEventHandler(async (event) => {
  if (!configured()) throw createError({ statusCode: 503, statusMessage: "Dashboard auth is not configured" });
  const body = await readBody(event);
  if (typeof body?.password !== "string" || !(await bcrypt.compare(body.password, passwordHash()))) {
    throw createError({ statusCode: 401, statusMessage: "Invalid password" });
  }
  setHeader(event, "set-cookie", cookieHeader(createSession()));
  return { authenticated: true };
});
