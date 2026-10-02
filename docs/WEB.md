# All-in-one web

From the repository root, with the GPU environment and trusted v25 model installed:

```bash
python replay_web_server.py --port 8000
```

Open http://localhost:8000/. For SSH, forward port 8000 in VS Code.
Select an OSZ, click **Upload map**, select a Taiko difficulty and NM/HR/DT/DTHR,
then click **Generate replay**. Completion opens the replay automatically on
the same player page. Saved replays appear in **Replay library**.
The server runs inference, not additional training. Closing the tab does not
stop generation; reopening restores the pending job.

Limits: 128 MB uploads, charts under ten minutes, one GPU job at a time.
The service also applies a global budget of 12 uploads per 15 minutes and a
120-second socket inactivity timeout. These are basic safeguards, not user authentication.
Upload only maps and music you have permission to share.

## Database and files

SQLite catalogue: `runs/replays.sqlite3`, configurable with `FLYTAIKO_REPLAY_DB`.
WAL mode supports worker writes and library reads. Binary replay assets/music
remain at `web-fly/public/demos/<dataset>/`; uploads/status/logs remain under
`runs/user_replays/<id>/`. The database stores metadata, not duplicate binaries.
Existing manifests/status files are imported at server startup. Library queries
use SQLite rather than scanning all files on every request.

Back up SQLite with its backup API, plus both file directories. For filesystem
backups stop the server and workers first; include WAL/SHM files if present.
The database alone cannot restore assets. No automatic deletion is enabled.

## Loading performance

Large JSON files use gzip sidecars generated once, refreshed when sources change.
Demo assets have a one-hour browser cache and retain integrity verification.
Recorded RGB bundles download only after **Load recorded model inputs** is
pressed, avoiding competition with initial music/trace loading. The catalogue
returns metadata only. Binary neuron traces still download in full; first-load
time depends on bandwidth and file size. Compression does not speed up GPU work.

## Vercel

Vercel serves static UI, not CUDA or SQLite. Deploy a durable GPU backend with
HTTPS separately. Set Vercel root to `web-fly`, use `vercel.json`, and configure
`FLYTAIKO_BACKEND_URL` to the backend HTTPS origin. On the backend configure
`FLYTAIKO_ALLOWED_ORIGINS=https://your-site.vercel.app` (comma-separated).
Before public exposure add reverse-proxy authentication, TLS, per-user rate
limits and timeouts. The one-job guard is not an authenticated multi-user queue.
Weights, uploads, music and generated assets are excluded from Git.

### Current preview deployment

Frontend: https://flytaiko.vercel.app/. It has been smoke-tested through the
public site: OSZ upload, v25 GPU inference, automatic playback and saved library.
The GPU service runs in tmux on port 8010 (`flytaiko_release_web`). A separate
tmux session (`flytaiko_public_tunnel`) keeps a Cloudflare Quick Tunnel alive.
This is a preview, not a durable production backend: Quick Tunnel addresses
change on restart and have no production uptime guarantee. Replace it with a
named tunnel or a stable HTTPS GPU host before advertising a permanent service.
When the backend URL changes, update `FLYTAIKO_BACKEND_URL` in Vercel and redeploy.

The public library uses a single full-map replay selector. Published manifests
load independently of historical job-status files. Internal smoke recordings
are hidden. Retired pre-v25 demos are archived outside the public directory;
NOCTASTRA versions are retained. SQLite tombstones prevent retired entries
from reappearing when legacy job statuses are imported at startup.

Git auto-deploy is not connected yet: grant the Vercel GitHub App access to the
private `NaHieu2005/FlyTaiko` repository and set the Git project root to `web-fly`.
The current deployment was made from that directory using Vercel CLI. Until Git
is connected, deploy from `web-fly` using `vercel deploy --prod --scope nahieu2005`.
Keep deployment protection for previews; the production alias is publicly readable.
