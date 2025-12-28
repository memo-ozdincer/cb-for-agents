# SLURM Job Scripts for Circuit Breakers Training

This directory contains SLURM batch scripts for training Circuit Breakers on different HPC clusters.

## Quick Start

### Fir (Alliance) - 1×H100 (Smoke Test)

1. **Setup** (on login node):
```bash
ssh fir.alliancecan.ca
cd $HOME/scratch
git clone <your-repo-url> harmful-agents-meta-dataset
cd harmful-agents-meta-dataset
bash scripts/hpc_setup_fir.sh
```

2. **Submit smoke test**:
```bash
cd $HOME/scratch/harmful-agents-meta-dataset
sbatch slurm/fir_cb_llama4_1xh100_debug.sbatch
```

3. **Monitor** (outputs go to `logs/`):
```bash
squeue -u $USER
tail -f logs/cb_llama4_1xh100_smoke_fir_JOBID.out
```

### Trillium (Alliance/SciNet) - 4×H100

1. **Setup** (on GPU login node):
```bash
ssh memoozd@trillium-gpu.scinet.utoronto.ca
cd $SCRATCH
git clone <your-repo-url> harmful-agents-meta-dataset
cd harmful-agents-meta-dataset
bash scripts/hpc_setup_trillium.sh
```

2. **Prepare data**:
```bash
source $SCRATCH/.venvs/cb_env/bin/activate
python scripts/ingest_cb_data.py
python scripts/prepare_cb_training_data.py --batch-size 16
```

3. **Edit account** in `slurm/trillium_cb_llama4_4xh100.sbatch`:
```bash
#SBATCH --account=def-YOURACCOUNT  # Replace with your RAP
```

4. **Submit job**:
```bash
cd $SCRATCH/harmful-agents-meta-dataset
sbatch slurm/trillium_cb_llama4_4xh100.sbatch
```

5. **Monitor**:
```bash
squeue -u $USER
tail -f cb_llama4_4xh100_JOBID.out
```

### CSLab ML Cluster - 4×H100

1. **Setup**:
```bash
ssh concerto  # or your CSLab ML cluster login
cd /mfs1/u/$USER
git clone <your-repo-url> harmful-agents-meta-dataset
cd harmful-agents-meta-dataset
bash scripts/hpc_setup.sh
```

2. **Prepare data**:
```bash
source /mfs1/u/$USER/.venvs/cb_env/bin/activate
python scripts/ingest_cb_data.py
python scripts/prepare_cb_training_data.py --batch-size 16
```

3. **Submit job**:
```bash
sbatch slurm/cslab_cb_llama4_4xh100.sbatch
```

4. **Monitor**:
```bash
squeue -u $USER
tail -f logs/cb_llama4_4xh100_JOBID.out
```

---

## Available Scripts

| Script | Cluster | GPUs | Time | Purpose |
|--------|---------|------|------|---------|
| `trillium_cb_llama4_4xh100.sbatch` | Trillium | 4×H100 | 6h | Full training |
| `trillium_cb_llama4_1xh100_debug.sbatch` | Trillium | 1×H100 | 30min | Quick test |
| `fir_cb_llama4_1xh100_debug.sbatch` | Fir | 1×H100 | 15min | Quick test |
| `cslab_cb_llama4_4xh100.sbatch` | CSLab ML | 4×H100 | 6h | Full training |

---

## Where Slurm Outputs Go

- If a script uses `#SBATCH --output=%x_%j.out` / `#SBATCH --error=%x_%j.err`, files are written in the directory you ran `sbatch` from.
- If a script uses `#SBATCH --output=logs/%x_%j.out` / `#SBATCH --error=logs/%x_%j.err`, files are written under `logs/` in the directory you ran `sbatch` from (the folder must exist before submission).

---

## Trillium-Specific Notes

### Important Trillium Constraints

1. **Submit from $SCRATCH**: Job outputs must go to `$SCRATCH` (not `$HOME` or `$PROJECT`).
   ```bash
   cd $SCRATCH/harmful-agents-meta-dataset
   sbatch slurm/trillium_cb_llama4_4xh100.sbatch
   ```

2. **GPU allocation**: Trillium allows `--gpus-per-node=1` or `--gpus-per-node=4` only (no 2 or 3).

3. **GPU login node**: Submit GPU jobs from `trillium-gpu.scinet.utoronto.ca`, not the CPU login node.

4. **SSH key auth**: Add your SSH public key to CCDB, then:
   ```bash
   ssh -i ~/.ssh/id_ed25519 memoozd@trillium-gpu.scinet.utoronto.ca
   ```

5. **Module system**: Jobs must load modules explicitly (handled in scripts):
   ```bash
   module load StdEnv/2023
   module load cuda/12.6
   module load python/3.11.5
   ```

### Trillium Workflow Example

```bash
# 1. SSH to GPU login
ssh -i ~/.ssh/id_ed25519 memoozd@trillium-gpu.scinet.utoronto.ca

# 2. Clone to $SCRATCH
cd $SCRATCH
git clone <repo> harmful-agents-meta-dataset
cd harmful-agents-meta-dataset

# 3. Setup environment
bash scripts/hpc_setup_trillium.sh

# 4. Login to services (interactive)
source $SCRATCH/.venvs/cb_env/bin/activate
huggingface-cli login
wandb login

# 5. Prepare data
python scripts/ingest_cb_data.py
python scripts/prepare_cb_training_data.py

# 6. Edit job script
nano slurm/trillium_cb_llama4_4xh100.sbatch
# Change: #SBATCH --account=def-XXXX

# 7. Test environment first
sbatch slurm/trillium_cb_llama4_1xh100_debug.sbatch
# Wait for completion, check output

# 8. Submit training
sbatch slurm/trillium_cb_llama4_4xh100.sbatch

# 9. Monitor
squeue -u $USER
tail -f cb_llama4_4xh100_*.out
```

---

## CSLab ML Cluster Notes

### Partitions

- Partition: `ml`
- Account: `ml`
- QOS: `ml`
- Nodes: concerto (8×H100)

### Requesting GPUs

Check with your cluster admin whether to use:
```bash
--gres=gpu:4          # Generic 4 GPUs
--gres=gpu:h100:4     # Specific H100 type
```

Current scripts use `--gres=gpu:4`.

### CSLab Workflow Example

```bash
# 1. SSH to cluster
ssh concerto

# 2. Clone to /mfs1 (fast storage)
cd /mfs1/u/$USER
git clone <repo> harmful-agents-meta-dataset
cd harmful-agents-meta-dataset

# 3. Setup
bash scripts/hpc_setup.sh

# 4. Login to services
source /mfs1/u/$USER/.venvs/cb_env/bin/activate
huggingface-cli login
wandb login

# 5. Prepare data
python scripts/ingest_cb_data.py
python scripts/prepare_cb_training_data.py

# 6. Submit
sbatch slurm/cslab_cb_llama4_4xh100.sbatch

# 7. Monitor
squeue -u $USER
tail -f logs/cb_llama4_4xh100_*.out
```

---

## Customizing Training

Edit the `accelerate launch` command in the job scripts:

```bash
accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  scripts/train_circuit_breaker.py \
    --preset llama-4-scout \           # Model preset
    --loss-weighting dual \             # Use dual coefficients
    --total-steps 300 \                 # Training steps
    --batch-size 4 \                    # Per-GPU batch size
    --gradient-accumulation 4 \         # Accumulation steps
    --cb-target-layers 12 24 36 \       # Layers for CB
    --output-dir outputs/my_run
```

Available presets:
- `llama-4-scout` - Llama-4-Scout-17B-16E (MoE)
- `llama-3-8b` - Llama-3-8B
- `mistral-7b` - Mistral-7B

---

## Troubleshooting

### Trillium: "Connection closed by remote host"

1. Ensure SSH key is uploaded to CCDB
2. Use explicit key:
   ```bash
   ssh -vvv -i ~/.ssh/id_ed25519 memoozd@trillium-gpu.scinet.utoronto.ca
   ```
3. Check verbose output for key rejection

### Job fails with "permission denied" on outputs

- **Trillium**: Submit from `$SCRATCH`, not `$HOME`
- **CSLab**: Ensure `/mfs1/u/$USER` is writable

### CUDA not available

- **Trillium**: Run on GPU login or in GPU job
- **Both**: Check `module list` shows CUDA module

### Out of memory

Reduce batch size or add gradient accumulation:
```bash
--batch-size 2 \
--gradient-accumulation 8
```

---

## References

- [Trillium Docs](https://docs.alliancecan.ca/wiki/Trillium)
- [Trillium Quickstart](https://docs.alliancecan.ca/wiki/Trillium_Quickstart)
- [Alliance GPU Guide](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm)
