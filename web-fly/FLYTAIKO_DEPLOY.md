# FlyTaiko public release

> Follow [current owner/cloud setup](../docs/OWNER_CLOUD.md). Vercel now also
> serves a Neon-backed catalogue API. Published assets use Blob; the older
> static-client notes below describe the previous deployment.

The Vercel site is a static client. It does not run MaleCNS inference, store
uploads, or publish model weights. GPU inference and generated demo files are
served by `python -m flytaiko.replay_web_server` on a separate, persistent GPU host.

## Frontend

Set `FLYTAIKO_BACKEND_URL` in the Vercel project to the public **HTTPS origin**
of the GPU service (no path or trailing slash). Use `web-fly/` as the project
root. `vercel.json` builds `dist-public/` from a short allowlist. The package
contains the Taiko player, uploader, Three.js and Flybody assets, not the
5+ GiB local `public/demos/` directory, beatmap library, uploaded music, or
the personal Koishi skin. Never commit those generated/user-provided files.

Local build check (Node 22+):

```sh
cd web-fly
FLYTAIKO_BACKEND_URL=http://localhost:8000 node scripts/build-public-release.mjs
python -m http.server 8765 --bind 127.0.0.1 --directory dist-public
```

## GPU service

The server must run on a persistent GPU host with its Python dependencies,
MaleCNS graph data, and the selected v25 checkpoint. Vercel Functions cannot
replace it: replay inference outlasts typical function lifetimes, and an .osz
upload often exceeds the Function body limit. Put the service behind HTTPS
and a reverse proxy with upload rate limits and abuse protection. Configure
`FLYTAIKO_ALLOWED_ORIGINS=https://YOUR-VERCEL-DOMAIN` before starting
`python -m flytaiko.replay_web_server`. Do not expose a development VS Code port-forward as
the permanent public GPU endpoint.

Only publish beatmaps, audio, and skins you have the right to redistribute.
Current generated replay URLs are bearer-by-link, **not private storage**.
Set a storage retention policy and monitor the 5 GiB disk reserve before
opening anonymous uploads to the internet.

## Release gate

1. Confirm the Git remote points to a repository you control, not the
   upstream `cobanov/fly-connectome-template` remote.
2. Confirm source attribution and licenses in `LICENSE`, `ATTRIBUTION.md`,
   and `THIRD_PARTY_NOTICES.md` remain in the release.
3. Test upload → NM replay → player on a different origin, and check the
   `Access-Control-Allow-Origin` response for the actual Vercel domain.
4. Check the Vercel deployment contains only `dist-public/` assets and no
   `public/demos/`, `public/beatmaps/`, checkpoint, or upload archive.

The training, checkpoint, GPU host, and generated replay storage are separate
from the frontend Git/Vercel release. Do not treat a successful static deploy
as evidence that public inference is operational.
