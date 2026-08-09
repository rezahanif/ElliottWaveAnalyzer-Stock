export default defineNitroConfig({
  srcDir: "server",
  compatibilityDate: "2026-08-06",
  runtimeConfig: {
    // Path to the shared SQLite DB written by the Python pipeline.
    // Overridable via NITRO_DB_PATH env var.
    dbPath: process.env.NITRO_DB_PATH || "../../data/predictions.db",
    // Root of the Python repo, used to resolve script_path when spawning jobs.
    repoRoot: process.env.NITRO_REPO_ROOT || "../..",
    // Required in deployment: bare python3 lacks pipeline dependencies.
    pythonBin: process.env.NITRO_PYTHON_BIN || "",
    jobTimeoutMs: process.env.NITRO_JOB_TIMEOUT_MS || "21600000",
    dashboardPasswordHash: process.env.DASHBOARD_PASSWORD_HASH || "",
    host: process.env.NITRO_HOST || "127.0.0.1",
  },
});
