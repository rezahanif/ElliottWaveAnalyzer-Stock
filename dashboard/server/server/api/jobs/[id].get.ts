import { useDb } from "../../utils/db";

export default defineEventHandler((event) => {
  const id = Number(getRouterParam(event, "id"));
  const db = useDb();

  const job = db.prepare(`SELECT * FROM jobs WHERE id = ?`).get(id);
  if (!job) {
    throw createError({ statusCode: 404, statusMessage: "Job not found" });
  }
  return job;
});
