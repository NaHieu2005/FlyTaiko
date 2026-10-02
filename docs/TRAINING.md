# Training v25

Follow the root README for the Python/CUDA environment and MaleCNS graph setup.
Run commands from the repository root. Songs, caches, dataset manifests and
weights are not distributed in Git. Load trusted PyTorch checkpoints only.

## Required dataset inputs

| Path | Content |
| --- | --- |
| `runs/dataset_metadata/base/split.json` | A:100 maps, C:50, normal validation/test |
| `runs/dataset_metadata/bonus/rows.json` | 22 bonus maps |
| `runs/dataset_metadata/low_sv/rows.json` | 30 additional slow-SV variants |
| same directory: `validation_rows.json`, `test_rows.json` | 8 slow-SV maps each |
| `runs/dataset_metadata/reference_config.json` | Reference sensory configuration |

Rows include parsed metadata, notes, timing and scroll information. Consult
`training.v25.train_full.prepare()` and `training.common.campaign_helpers.entries()` for the schema.
A fresh checkout cannot train until these manifests and the graph are provided.
`models/v25/config.json` is a reference; preserve verified splits and metadata.

## Run

```bash
mkdir -p runs
python -m training.v25.train_full --help
python -m training.v25.train_full --preflight-only
python -u -m training.v25.train_full > runs/malecns-v25-full.log 2>&1
```

For unattended execution use tmux on the remote GPU server. The campaign builds
new sensory caches, trains A then C, selects checkpoints with validation and
evaluates held-out maps. Upper limits are 12 A and 8 C epochs, with stale-metric
early stopping. Reserve tens of GB of disk space and GPU headroom. Never mix
caches from different renderers/configs. Source hashes and song-disjoint checks
guard compatibility and leakage. Training uses clips, not every full-song frame.

Monitor normal and slow-SV Great/Good/Miss, false hits, signed timing error and
spinner completion, not loss alone. Do not tune using test results. Inspect
`runs/malecns_v25_full/result.json` and the selected checkpoint before publishing.
Low loss does not guarantee SS or unseen-map generalization.

On the maintained workstation, the completed original v25 run is preserved at
`runs/v25_completed_training`; its caches were retired. The active model is
still `models/v25/policy.pt`. Refactored source hashes intentionally prevent
resuming old optimizer/cache state under changed code. New preflight metadata
uses `runs/malecns_v25_full`; do not bypass compatibility checks to resume.

## Publish

Place the selected trusted checkpoint at `models/v25/policy.pt` and its exact
matching config at `models/v25/config.json`, or set `FLYTAIKO_MODEL_DIR`.
`flytaiko.replay_models.load_v25()` rejects config/checkpoint mismatches. Restart the GPU
service after changing weights. Existing replays remain historical recordings.
See [WEB.md](WEB.md) for upload and playback instructions.

This is an experimental graded-rate controller, not a validated biological brain
simulation or a controller for the actual osu! client.
