# Quick Start Guide - After Cache Fix

## TL;DR - Run These Commands

```bash
# 1. On login node (has internet):
cd /project/def-zhijing/memoozd/harmful-agents-meta-dataset
export HF_TOKEN=hf_xxxxxxxxxxxxx  # Your token
bash slurm/Trillium/prefetch_models.sh

# 2. Verify cache:
ls -1 /scratch/memoozd/cb_cache/hf/hub/models--*

# 3. Submit jobs:
sbatch slurm/Trillium/trillium_mvp_generate_ds.sbatch
# Then after it finishes:
sbatch slurm/Trillium/trillium_mvp_generate_dr.sbatch
sbatch slurm/Trillium/trillium_mvp_create_eval.sbatch
sbatch slurm/Trillium/trillium_mvp_validate.sbatch
sbatch slurm/Trillium/trillium_mvp_train.sbatch
sbatch slurm/Trillium/trillium_mvp_eval.sbatch
```

## What Changed?

**One sentence**: Removed deprecated `TRANSFORMERS_CACHE`, unified all caching to `HF_HOME/hub`, and switched prefetch to use `snapshot_download()`.

**Files modified**:
- ✅ `slurm/Trillium/prefetch_models.sh` - Now uses `snapshot_download()`
- ✅ `slurm/Trillium/trillium_mvp_generate_ds.sbatch` - Clean cache config
- ✅ `slurm/Trillium/trillium_mvp_generate_dr.sbatch` - Clean cache config
- ✅ `slurm/Trillium/trillium_mvp_create_eval.sbatch` - Clean cache config
- ✅ `slurm/Trillium/trillium_mvp_validate.sbatch` - Clean cache config
- ✅ `slurm/Trillium/trillium_mvp_train.sbatch` - Clean cache config
- ✅ `slurm/Trillium/trillium_mvp_eval.sbatch` - Clean cache config

## Standard Cache Config (Now Used Everywhere)

```bash
CACHE_ROOT=/scratch/memoozd/cb_cache
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# NO TRANSFORMERS_CACHE!
```

## Debugging

If a job fails with cache errors, check the log:

```bash
tail -50 /scratch/memoozd/logs/mvp_ds_gen_*.out | grep -A 5 "Cache Configuration"
```

Should see:
```
Cache Configuration:
  HF_HOME: /scratch/memoozd/cb_cache/hf
  HF_HUB_CACHE: /scratch/memoozd/cb_cache/hf/hub
  HF_HUB_OFFLINE: 1

Hub cache contents:
models--meta-llama--Llama-3.1-8B-Instruct
models--mlabonne--Meta-Llama-3.1-8B-Instruct-abliterated
```

If you see `(no models cached yet)`, re-run `prefetch_models.sh`.

## Clean Slate (Optional)

To start fresh:

```bash
rm -rf /scratch/memoozd/cb_cache/hf
bash slurm/Trillium/prefetch_models.sh
```

---

**See `CACHE_FIX_SUMMARY.md` for detailed explanation.**
