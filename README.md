# FlyTaiko

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
.venv/bin/python prepare_malecns.py
.venv/bin/python malecns_system.py
```

Place the trusted selected v25 checkpoint at `models/v25/policy.pt`, alongside
its bundled `config.json`. Set `FLYTAIKO_MODEL_DIR` to use another directory
containing those two files. Model weights, downloaded graph data, training
caches, uploaded archives, and generated demos are supplied separately.

```sh
.venv/bin/python replay_web_server.py --bind 127.0.0.1 --port 8000
```

Open `http://localhost:8000/create-replay.html`, upload an `.osz`, choose a Taiko
difficulty and NM/HR/DT/DTHR, then create the replay. Jobs run asynchronously
and the finished demo appears in the replay library. Recorded metrics, input
images and neuron traces are shown with the original recorded key timing.

## Public website on Vercel

Import this repository into Vercel with Root Directory `web-fly`. Set
`FLYTAIKO_BACKEND_URL` to your GPU service's HTTPS origin. The bundled
`vercel.json` builds a static package of approximately 4 MB. On the GPU host,
set `FLYTAIKO_ALLOWED_ORIGINS` to the frontend origin; multiple origins can be
comma-separated. Uploads and demo data travel directly to the GPU service.
See [deployment instructions](web-fly/FLYTAIKO_DEPLOY.md).

The GPU backend must stay online to generate replays. The Vercel site serves
the frontend; it does not execute CUDA inference.

## Source layout

- `replay_web_server.py`, `replay_osz_worker.py`, `replay_models.py`: upload API,
  replay job and selected-model loading.
- `malecns_system.py`, `measured_rate_system.py`, `sensory_readout.py`,
  `sensory_temporal_policy.py`: measured graph, graded activity and causal GRU.
- `visual_taiko.py`, `highres_taiko.py`, `motor_modes.py`, `taiko/`: renderer,
  native objects, judgments and action interface.
- `prepare_malecns_taiko_demo.py`: checks and publishes recorded gameplay/trace.
- `train_v25_full.py`: fresh-cache A/C training with slow-SV and spinner
  validation. Reproduction also requires the original song-disjoint source
  manifests referenced by the training script; those datasets are separate.
- `web-fly/public/`: current Taiko player, uploader, neural display and 3D scene.

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
