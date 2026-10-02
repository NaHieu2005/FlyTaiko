# Training versions

Run module entry points from the repository root, not individual files.

| Directory | Purpose | Entry point |
| --- | --- | --- |
| `v25/` | Current fresh-cache A/C campaign | `python -m training.v25.train_full` |
| `v23/` | Historical sensory-memory campaign and v23/v24 spinner finetuning | `python -m training.v23.train_full`, `python -m training.v23.spinner_finetune` |
| `common/` | Parallel caching, fingerprints and native split refresh | `python -m training.common.cache_parallel` |
| `legacy/` | Early campaign implementation required by shared code | `python -m training.legacy.train_campaign` |

v25 currently imports reusable clip/training helpers from v23. Therefore v23
is retained, not deleted. Moving a file does not mean its methods are unused.
Training source hashes now name package-relative paths. Existing checkpoints
and their matching configuration metadata are not rewritten or retrained.
Old cache fingerprints remain historical and should not be silently reused
under a changed source layout. See [training prerequisites](../docs/TRAINING.md).
