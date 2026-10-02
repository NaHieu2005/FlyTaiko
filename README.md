# FlyTaiko

## Current public deployment

Public playback at https://flytaiko.vercel.app/ now uses Neon Postgres and
Vercel Blob, independently of the workstation. Only the owner can upload and
generate replays, using **Owner sign in**. See [current setup and workflow](docs/OWNER_CLOUD.md);
the GPU backend is needed only for generation. Older local-service notes below
describe workstation recovery storage, not the public playback architecture.

Taiko replay inference with measured MaleCNS connectivity, a causal sensory
memory policy, and a browser player with recorded neural activity and a 3D fly.
The selected release model is **v25 Phase C, epoch 4**. Validation improves
circle accuracy and spinner coverage, but false hits remain a known limitation.
Evaluation results and model configuration are in `models/v25/`.

## Run the GPU service

Use Python 3.11 and a compatible NVIDIA CUDA GPU. The tested environment uses
PyTorch 2.5.1/CUDA 12.4; install the tested dependencies in a virtual environment:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-gpu-cu124.lock
.venv/bin/python -m flytaiko.prepare_malecns
.venv/bin/python -m flytaiko.malecns_system
```

Place the trusted selected v25 checkpoint at `models/v25/policy.pt`, alongside
its bundled `config.json`. Set `FLYTAIKO_MODEL_DIR` to use another directory
containing those two files. Model weights, downloaded graph data, training
caches, uploaded archives, and generated demos are supplied separately.

```sh
.venv/bin/python -m flytaiko.replay_web_server --bind 127.0.0.1 --port 8000
```

Open `http://localhost:8000/` for local playback. For owner uploads and cloud
publication, use the authenticated public portal and worker environment in
[OWNER_CLOUD.md](docs/OWNER_CLOUD.md).

## Public website on Vercel

Import this repository into Vercel with Root Directory `web-fly`. Set
`FLYTAIKO_BACKEND_URL` to your GPU service's HTTPS origin. The bundled
`vercel.json` builds a static package of approximately 4 MB. On the GPU host,
set `FLYTAIKO_ALLOWED_ORIGINS` to the frontend origin; multiple origins can be
comma-separated. Owner uploads travel to the GPU; published files load from Blob.
See [deployment instructions](web-fly/FLYTAIKO_DEPLOY.md).

The GPU backend must stay online to generate replays, but not to view published
ones. Vercel does not execute CUDA inference. A future independent GPU-host
container recipe is in [CLOUD.md](docs/CLOUD.md).

## Source layout

```text
flytaiko/           Runtime, measured brain, policy, rendering and replay API
training/
  v25/             Current fresh-cache A/C campaign
  common/          Shared campaign helpers, cache workers and split tools
taiko/             Native parser, objects and environment
flyconnectome/     Early connectome simulation dependencies
models/v25/        Selected model configuration and evaluation metadata
web-fly/           Vercel frontend and browser player
deploy/cloud/      Independent GPU-host container recipe
docs/              Training, web and cloud migration guides
tests/             Logic, export and catalogue checks
```

Run `python -m training.v25.train_full` from the repository root. Training
requires the original song-disjoint dataset manifests, provided separately.
Reusable methods are consolidated in `training/common`; old version directories
are removed. See [training layout](training/README.md).

## Checks

```sh
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q tests
cd web-fly
FLYTAIKO_BACKEND_URL=http://localhost:8000 node scripts/build-public-release.mjs
```

## Credits and model scope

Built with [fly-connectome-template](https://github.com/cobanov/fly-connectome-template)
by [Mert Cobanov](https://github.com/cobanov). This project modifies the template
for Taiko playback, recorded activity, UI overlays and the fly scene. Its
[license](web-fly/LICENSE) and [attribution requirements](web-fly/ATTRIBUTION.md)
are retained. Flybody is Apache-2.0; see the bundled asset notices.

MaleCNS v1.0 connectivity and cell IDs come from FlyEM / HHMI Janelia,
University of Cambridge, MRC Laboratory of Molecular Biology and Google
Research, under CC BY 4.0. See [the dataset source](https://male-cns.janelia.org/download/).
Neuron dynamics, visual sampling and the trained decoder are engineering
models. The activity display uses recorded model values and does not claim
measured biological firing rates. The current release generates offline
replays; it does not control an installed osu! client.
# Documentation

- [Training and dataset prerequisites](docs/TRAINING.md)
- [All-in-one web, replay database, uploads and deployment](docs/WEB.md)
- [Removing workstation dependence](docs/CLOUD.md)
