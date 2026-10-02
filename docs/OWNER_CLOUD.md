# Owner uploads and independent playback

Open https://flytaiko.vercel.app/. Public replay playback uses **Neon Postgres**
for the catalogue and **Vercel Blob** for files, music and recorded activity.
Viewing published replays does not need the workstation. Generation still does.

## Owner workflow

Click **Owner sign in**. Use the installation password stored privately in
`runs/owner-password.txt`, not your GitHub password. Upload an OSZ, select a
Taiko difficulty and NM/HR/DT/DTHR, then **Generate replay**. All referenced
assets upload before the public catalogue entry is committed. Closing the tab
does not stop generation; signing in again restores a pending job.

Only the owner can upload, generate or read job status. Login uses a Secure,
HttpOnly, SameSite=Strict cookie; GPU requests use short-lived signed tokens.
Publication uses a separate machine secret. This is a single-owner password
gate, not a multi-user account system. Use HTTPS and a strong password.

## Installation

Use Node 22+ alongside the Python/CUDA environment. Link the intended Vercel
project with root `web-fly`, attach a public Blob store and Neon Postgres:

```sh
cd web-fly
npm ci
vercel env pull .env.local --environment development
node --env-file=.env.local scripts/setup-database.mjs
node --env-file=.env.local scripts/setup-owner.mjs
vercel deploy --prod --scope nahieu2005
```

The setup script targets the current scope/domain; edit these for another
installation. It reuses `runs/owner-secrets.json` and writes mode-600
`runs/owner-password.txt` and `runs/cloud.env`. Never commit or serve them.
Vercel requires `DATABASE_URL`, `BLOB_READ_WRITE_TOKEN`, `FLYTAIKO_AUTH_SECRET`,
`FLYTAIKO_PUBLISH_KEY`, `FLYTAIKO_ADMIN_PASSWORD_SHA256`,
`FLYTAIKO_ASSET_ORIGIN`, `FLYTAIKO_BACKEND_URL` and optionally
`FLYTAIKO_ADMIN_ORIGIN`. Only asset origin/backend URL are public configuration.

Start the worker from the repository root with Node on PATH:

```sh
set -a
source runs/cloud.env
set +a
export FLYTAIKO_ALLOWED_ORIGINS=https://flytaiko.vercel.app
python -m flytaiko.replay_web_server --port 8010
```

Current tmux sessions: `flytaiko_release_web` and `flytaiko_public_tunnel`.
Quick Tunnel addresses change on restart; update the backend URL and redeploy.
Git auto-deploy is not connected. Container migration also needs Node and
installed web dependencies for publication; that container remains unvalidated.

## Storage, speed and recovery

### Backgrounds and personal skins

Beatmap backgrounds are extracted from OSZ Events and published with each new
replay. **Background brightness** controls the same canvas used by the 2D
player and the monitor in the 3D scene; the setting is remembered locally.

**Import skin** accepts `.osk`, `.zip` or a skin folder. Selected Taiko PNG
notes/overlays, lane, roll, spinner and OGG/WAV/MP3 hitsounds are stored in
browser IndexedDB, not uploaded to the public library. `@2x` is preferred;
missing optional assets fall back to the default presentation. Hitcircle is
required. Skin notes use the existing fixed gameplay geometry, not arbitrary
image dimensions. This is a Taiko subset, not complete osu! skin.ini/animation
compatibility. Brightness and skins affect presentation only, not model input
or replay judgments. Archive limit is 64 MB, selected asset limit 40 MB and
individual asset limit 8 MB. Clearing website data removes imported skins.

The player and 3D monitor use 16:9 geometry. Imported drum inner/outer halves
show independent D/F/J/K inputs beside the judgment target. Re-import older
saved skins to include newly supported assets. Re-importing the same name replaces the existing saved
skin and removes duplicate entries (case-insensitive, ignoring edge spaces).
An invalid import does not delete the saved skin. Note overlays support static
and two-frame variants. Kiai intervals come from native TimingPoints; lane glow
uses the skin's glow asset when present. Optional finish/whistle, combo-break
and spinner-completion samples are supported alongside Don/Kat and roll hits.
Missing samples use the existing fallback where applicable. This is not full
osu! sample-set/custom per-object sample or storyboard parity.
Don/Kat imports support `taiko-drum-hit*`, `taiko-normal-hit*` and
`taiko-soft-hit*`, never unprefixed standard `normal-hit*`.
Assets in a skin's `taiko/` subfolder take priority. Re-import to replace any
previously saved standard-sample fallback. Drum idle artwork is shown once;
individual skin drum halves light only when their recorded key is pressed.
Overlay display is static until native animation timing is supported; there
is no synthetic combo/BPM frame switching. Import reports decoded sounds,
missing Don/Kat fallbacks and unsupported/corrupt audio independently.

Neon stores manifests/metrics. Blob assets use SHA-256 content-addressed names,
deduplication and one-year immutable caching. Gameplay loads first, traces
follow, and recorded images download only on request. First loads still depend
on bandwidth and file size; cloud storage is not instantaneous playback.

Local SQLite, `runs/user_replays/` and `web-fly/public/demos/` remain recovery
data. Back them up alongside cloud storage. Retry publication with:

```sh
cd web-fly
node --env-file=../runs/cloud.env scripts/publish-replay.mjs DATASET
```

Migrate sequentially to avoid concurrent writes of identical assets. Smoke
recordings are hidden. Retained NOCTASTRA versions are public.
Limits remain 128 MB uploads, maps under ten minutes and one GPU job at a time.
Publish only maps/music you may redistribute. Blob files are public by URL.
Monitor storage/bandwidth billing; free database provisioning does not make
all file storage/traffic free or unlimited.
