# Training versions

Run module entry points from the repository root, not individual files.

| Directory | Purpose | Entry point |
| --- | --- | --- |
| `v25/` | Current fresh-cache A/C campaign | `python -m training.v25.train_full` |
| `common/` | Parallel caching, fingerprints and native split refresh | `python -m training.common.cache_parallel` |

v25 imports shared clip/training helpers from `common/campaign_helpers.py`,
`common/long_object_helpers.py` and `common/short_campaign_helpers.py`.
Historical v23/legacy directories and their entry points are removed.
Training source hashes now name package-relative paths. Existing checkpoints
and their matching configuration metadata are not rewritten or retrained.
Old cache fingerprints remain historical and should not be silently reused
under a changed source layout. See [training prerequisites](../docs/TRAINING.md).
