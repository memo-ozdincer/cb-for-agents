#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=probe_agentdojo_ingest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# ============================================================================
# TEMP QUICK PROBE: AgentDojo raw format -> ingest output
#
# Goal:
# - Print RAW AgentDojo examples (keys, where messages live, tool_calls presence)
# - Run ingest_agentdojo_splits.py on a tiny subset
# - Print PROCESSED outputs (harmful/retain) and <|python_tag|> coverage
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_quick_probe_agentdojo_ingest_TEMP.sbatch
# ============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load python/3.11.5
module list

echo "=== Venv activation ==="
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"

RUN_DIR="$SCRATCH_DIR/stage2_probe_agentdojo/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

echo ""
echo "========================================"
echo "1) RAW INPUT PROBE"
echo "========================================"

INPUT_DIR_DEFAULT="$REPO_DIR/data/agent_dojo"
INPUT_DIR="${INPUT_DIR:-$INPUT_DIR_DEFAULT}"

# If you have the real AgentDojo traces elsewhere, override here:
# INPUT_DIR="$SCRATCH_DIR/agent_dojo"

echo "Input dir: $INPUT_DIR"
ls -lah "$INPUT_DIR" | head -n 50

export INPUT_DIR

echo ""
echo "--- RAW JSONL SCHEMA PROBE (first ~30 lines of up to 3 files) ---"
python - << 'PY'
import json
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/project/def-zhijing/memoozd/harmful-agents-meta-dataset/data/agent_dojo"))

paths = sorted([p for p in INPUT_DIR.glob("*.jsonl") if p.is_file()])
print("found_files", len(paths))
for p in paths[:3]:
    print("\nFILE", p.name, "size", p.stat().st_size)
    key_counts = {}
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print("  JSONDecodeError:", str(e)[:120])
                continue
            n += 1
            for k in r.keys():
                key_counts[k] = key_counts.get(k, 0) + 1
            if n <= 2:
                print("  EXAMPLE_LINE", n)
                print("    top_keys:", sorted(r.keys()))
                for cand in ("messages", "trace", "data", "record", "example"):
                    if cand in r:
                        v = r[cand]
                        t = type(v).__name__
                        extra = ""
                        if isinstance(v, list):
                            extra = f" len={len(v)}"
                        if isinstance(v, dict):
                            extra = f" keys={sorted(list(v.keys()))[:15]}"
                        print(f"    {cand}: {t}{extra}")
                # Try to locate messages list anywhere shallow
                def find_messages_shallow(obj, prefix=""):
                    out = []
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k == "messages" and isinstance(v, list):
                                out.append(prefix + ".messages")
                            if isinstance(v, dict):
                                out.extend(find_messages_shallow(v, prefix + "." + k))
                    return out
                paths_found = find_messages_shallow(r, prefix="$")
                print("    messages_paths:", paths_found[:10])
            if n >= 30:
                break
    print("  lines_scanned", n)
    print("  key_counts", dict(sorted(key_counts.items(), key=lambda x: -x[1])[:20]))
PY

echo ""
echo "========================================"
echo "2) RUN INGEST (SMALL)"
echo "========================================"

OUT_HARMFUL="$RUN_DIR/agentdojo_failures_probe.jsonl"
OUT_RETAIN="$RUN_DIR/agentdojo_resisted_probe.jsonl"

python scripts/cb_data_generation/ingest_agentdojo_splits.py \
  --input-dir "$INPUT_DIR" \
  --file-pattern "agentdojo-*.jsonl" \
  --output-harmful "$OUT_HARMFUL" \
  --output-retain "$OUT_RETAIN" \
  --max-each 50

echo ""
echo "Ingest outputs:"
echo "  harmful: $OUT_HARMFUL ($(wc -l < "$OUT_HARMFUL" 2>/dev/null || echo 0) lines)"
echo "  retain:  $OUT_RETAIN ($(wc -l < "$OUT_RETAIN" 2>/dev/null || echo 0) lines)"

echo ""
echo "========================================"
echo "3) PROCESSED OUTPUT PROBE (PRINT SAMPLES)"
echo "========================================"

python - << 'PY'
import json
import os
from collections import Counter
from pathlib import Path

job_id = os.environ.get("SLURM_JOB_ID", "")
run_dir = Path("/scratch/memoozd") / "stage2_probe_agentdojo" / job_id
harmful_path = run_dir / "agentdojo_failures_probe.jsonl"
retain_path = run_dir / "agentdojo_resisted_probe.jsonl"

def read_jsonl(path: Path, limit: int | None = None):
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out

harmful = read_jsonl(harmful_path)
retain = read_jsonl(retain_path)
print("run_dir:", run_dir)
print("harmful_lines:", len(harmful), "retain_lines:", len(retain))

def tag_stats(samples):
    n = len(samples)
    with_tag = sum(1 for s in samples if "<|python_tag|>" in (s.get("assistant_raw") or ""))
    avg_len = sum(len((s.get("assistant_raw") or "")) for s in samples) / max(1, n)
    return {"n": n, "with_tag": with_tag, "pct": 100*with_tag/max(1,n), "avg_assistant_raw_chars": avg_len}

print("harmful_tag_stats:", tag_stats(harmful))
print("retain_tag_stats:", tag_stats(retain))

def print_samples(title, samples, k=5):
    print("\n---", title, f"(showing {min(k, len(samples))}/{len(samples)})", "---")
    for i, s in enumerate(samples[:k]):
        aid = s.get("id")
        src = (s.get("metadata") or {}).get("source")
        labels = s.get("labels") or {}
        ar = s.get("assistant_raw") or ""
        has_tag = "<|python_tag|>" in ar
        has_tool_calls = (s.get("metadata") or {}).get("has_tool_calls")
        obs = labels.get("observed_tool")
        print(f"[{i}] id={aid} source={src} split={labels.get('split')} observed_tool={obs}")
        print(f"    assistant_raw_chars={len(ar)} has_tag={has_tag} metadata.has_tool_calls={has_tool_calls}")
        print("    assistant_raw_head:", ar[:240].replace("\n", " "))
        msgs = s.get("messages") or []
        if msgs:
            last = msgs[-1]
            print(f"    last_context_msg: role={last.get('role')} head={str(last.get('content') or '')[:140].replace('\n',' ')}")

print_samples("HARMFUL SAMPLES", harmful, k=5)
print_samples("RETAIN SAMPLES", retain, k=5)

# Tool distribution
harmful_tools = Counter((s.get("labels") or {}).get("observed_tool") or "<none>" for s in harmful)
retain_tools = Counter((s.get("labels") or {}).get("observed_tool") or "<none>" for s in retain)
print("\nobserved_tool_top_harmful:", harmful_tools.most_common(10))
print("observed_tool_top_retain:", retain_tools.most_common(10))

# Diagnose why no tag (harmful)
no_tag = [s for s in harmful if "<|python_tag|>" not in (s.get("assistant_raw") or "")]
print("\nHARMFUL_NO_TAG_COUNT:", len(no_tag))
if no_tag:
    ex = no_tag[0]
    print("HARMFUL_NO_TAG_EXAMPLE id:", ex.get("id"))
    print("assistant_raw_head:", (ex.get("assistant_raw") or "")[:400].replace("\n", " "))
    print("metadata_keys:", sorted((ex.get("metadata") or {}).keys()))
    print("labels:", ex.get("labels"))
PY

echo ""
echo "========================================"
echo "DONE. Logs at: /scratch/memoozd/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "Probe artifacts in: $RUN_DIR"
echo "========================================"

echo ""
echo "========================================"
echo "4) MERGE PROBE (ONLY THESE OUTPUTS)"
echo "========================================"

MERGE_HARMFUL_DIR="$RUN_DIR/harmful"
MERGE_RETAIN_DIR="$RUN_DIR/retain"
mkdir -p "$MERGE_HARMFUL_DIR" "$MERGE_RETAIN_DIR"

# Put them into directories the merge script expects
cp -f "$OUT_HARMFUL" "$MERGE_HARMFUL_DIR/agentdojo_failures.jsonl"
cp -f "$OUT_RETAIN" "$MERGE_RETAIN_DIR/agentdojo_resisted.jsonl"

MERGED_TRAIN="$RUN_DIR/merged_train.jsonl"
MERGED_EVAL="$RUN_DIR/merged_eval.jsonl"

python scripts/cb_data_generation/merge_stage2_data.py \
  --harmful-dir "$MERGE_HARMFUL_DIR" \
  --retain-dir "$MERGE_RETAIN_DIR" \
  --output "$MERGED_TRAIN" \
  --eval-output "$MERGED_EVAL" \
  --eval-fraction 0.2 \
  --dr-ratio 1.0 \
  --no-shuffle

echo "Merged outputs:"
echo "  train: $MERGED_TRAIN ($(wc -l < "$MERGED_TRAIN" 2>/dev/null || echo 0) lines)"
echo "  eval:  $MERGED_EVAL ($(wc -l < "$MERGED_EVAL" 2>/dev/null || echo 0) lines)"

python - << 'PY'
import json
from pathlib import Path

train = Path("/scratch/memoozd") / "stage2_probe_agentdojo" / Path(__import__("os").environ.get("SLURM_JOB_ID","")) / "merged_train.jsonl"
def read(path):
    out=[]
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
xs = read(train)
harm = [s for s in xs if (s.get('labels') or {}).get('split')=='harmful']
with_tag = sum(1 for s in harm if '<|python_tag|>' in (s.get('assistant_raw') or ''))
print('merged_train_total', len(xs))
print('merged_train_harmful', len(harm), 'harmful_with_tag', with_tag, f"({100*with_tag/max(1,len(harm)):.1f}%)")
print('first_3_harmful:')
for s in harm[:3]:
    ar = s.get('assistant_raw') or ''
    print('  id', s.get('id'), 'has_tag', '<|python_tag|>' in ar, 'assistant_raw_head', ar[:160].replace('\n',' '))
PY
