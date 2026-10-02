# Independence from the workstation

The current website still uses the workstation API via a temporary tunnel.
Making the GitHub repository public does not move model execution or replay
storage to GitHub or Vercel. No paid resources have been provisioned.

Vercel does not provide native GPU execution for this CUDA workload. See
[Vercel's serverless GPU explanation](https://vercel.com/i/what-is-serverless-gpu).
The current graph is about 1.2 GB, in addition to model weights, Python/CUDA
dependencies, uploads and recorded traces. A browser-only port would require
a different inference implementation and performance validation; publishing
the GRU alone would skip the measured brain and would not be the same model.

## Cloud architecture

- Vercel: frontend, public URL and static 3D assets.
- An independent GPU cloud host: bounded upload API and CUDA inference worker.
- Durable cloud volume: graph, model, SQLite catalogue, uploads and demos.
- Stable HTTPS endpoint/reverse proxy: authentication and per-user limits.

This removes reliance on the workstation while preserving the neural pipeline.
Object storage/CDN and a managed job database can replace the cloud volume
later, but require storage-provider credentials and an explicit budget. Current
SQLite is supported on one persistent host; do not put it on ephemeral Vercel
functions or launch replicas that share an ordinary SQLite file over a network.

## Prepared container

`deploy/cloud/Dockerfile` packages the runtime. It deliberately excludes
weights, graph, uploads and recorded files. `deploy/cloud/compose.yaml` mounts
these separately and binds the API to localhost for a TLS reverse proxy.
Docker, NVIDIA Container Toolkit and a compatible NVIDIA driver are required
on the chosen cloud host.

1. Clone the public repo onto the GPU host.
2. Provision the trusted matching checkpoint/config in `models/v25/`, the
   MaleCNS graph in `connectome_data/`, and persistent `runs/` and demo storage.
3. Copy the existing SQLite catalogue **using its backup API** and the complete
   demo files. Resolve workstation symlinks into actual files; copying symlinks
   alone does not transfer data. Validate all referenced hashes after copying.
4. From the repo root, run `docker compose -f deploy/cloud/compose.yaml up -d --build`.
5. Put the API behind stable HTTPS, configure allowed frontend origins and
   authentication/rate limits, and perform a full upload/inference/playback test.
6. Change `FLYTAIKO_BACKEND_URL` in Vercel to the cloud origin and redeploy.
7. Verify playback/generation with the workstation API and tunnel stopped.
   Only then shut down the old service; keep the migration backup.

The container recipe is prepared but is not yet cloud-deployed or tested with
Docker: this workstation account cannot access the Docker daemon and lacks
the Compose plugin. Local package imports and CUDA smoke inference passed.

## Playback-only alternative

Existing immutable demos can be served from cloud object storage/CDN without
any GPU. This would make viewing independent but cannot generate new replays.
Audio and large RGB/trace bundles are separate assets; publishing only the
manifest is insufficient. Publish only assets you may redistribute. Hosting
them in Git source is not a substitute for durable upload storage.
