# Elliott Wave Dashboard Operations

## Security defaults

Dashboard server binds to `127.0.0.1` by default. Override only deliberately:

```bash
NITRO_HOST=127.0.0.1
DASHBOARD_PASSWORD_HASH='$2b$12$...'
NITRO_DB_PATH=../../data/predictions.db
NITRO_REPO_ROOT=../..
NITRO_PYTHON_BIN=/home/rezaserver/miniconda3/envs/elliott/bin/python
```

Generate bcrypt hash outside repository:

```bash
node -e 'require("bcryptjs").hash(process.argv[1], 12).then(console.log)' 'choose-password'
```

Never commit plaintext password, token, or hash to repository. Put values in systemd `EnvironmentFile` with mode `0600`.

## Tailscale edge access

Recommended topology: Nitro loopback listener + Tailscale-only access. Do not expose port 3000 directly to WAN.

Install/configure Tailscale on host, then authenticate node through your tailnet. Example systemd unit, matching repo service style:

```ini
[Unit]
Description=Elliott Wave Nitro Dashboard
After=network.target tailscaled.service
Requires=tailscaled.service

[Service]
Type=simple
User=rezaserver
WorkingDirectory=/home/rezaserver/ElliottWaveAnalyzer/dashboard/server
EnvironmentFile=/home/rezaserver/.config/elliott/dashboard.env
ExecStart=/usr/bin/env npm run preview
Restart=always
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

`dashboard.env` must include `DASHBOARD_PASSWORD_HASH`, `NITRO_HOST=127.0.0.1`, database/repo paths, and `NITRO_PYTHON_BIN`. Access from a tailnet client through a reverse proxy or explicit local forwarding. Tailscale Serve is preferred over direct port publication.

## Verification

1. `ss -ltnp | grep ':3000'` — expect `127.0.0.1:3000`, not `0.0.0.0:3000`.
2. Local unauthenticated `/api/assets` — expect `401`.
3. Login, retain `HttpOnly` session cookie, then read assets and open SSE.
4. Unauthenticated `POST /api/jobs` — expect `401`.
5. From cellular network, direct host-IP:3000 must fail to connect.
6. Through Tailscale path, login must be required and authenticated read/job routes must work.

## Legacy diagnostics service

`scripts/elliott-web.service` remains separate for `data/diagnostics`. Keep it only if diagnostics access is still needed. If enabled, bind its Python HTTP server to loopback or expose it through the same private Tailscale path; never publish port 8080 directly.

## Secret rotation and git history

The historical Telegram bot token is compromised. Rotate it with `@BotFather`, set `STOCK_TELEGRAM_BOT_TOKEN` in runtime environment, and keep `config/stock.yaml` empty. History rewrite with `git filter-repo` changes all commit IDs and requires coordinated force-push; perform only after all clones/users are notified.

CORS remains disabled by default because SPA/API should be same-origin behind deployment edge.
