#!/bin/bash
#SBATCH --job-name=etl_a_llmail
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# ETL_A: Convert Raw Data to B1 Skeleton Traces
#
# Processes:
#   - Fujitsu B4 (orchestrator attacks)
#   - AgentDojo (multi-turn agent attacks)  
#   - LLMail-Inject (email injection attacks)
#
# Output: B1 skeleton traces ready for generate_completions.py
#
# Usage:
#   sbatch slurm/pipeline/01_etl_a.sbatch
#
# Or selective processing:
#   SKIP_FUJITSU=true sbatch slurm/pipeline/01_etl_a.sbatch
#   SKIP_AGENTDOJO=true sbatch slurm/pipeline/01_etl_a.sbatch
#   ONLY_LLMAIL=true sbatch slurm/pipeline/01_etl_a.sbatch
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

# Load modules
module --force purge || true
module load StdEnv/2023
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR="$CB_SCRATCH/data/traces"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CB_SCRATCH/logs"

# Selective processing flags
SKIP_FUJITSU="${SKIP_FUJITSU:-false}"
SKIP_AGENTDOJO="${SKIP_AGENTDOJO:-false}"
SKIP_LLMAIL="${SKIP_LLMAIL:-false}"
ONLY_LLMAIL="${ONLY_LLMAIL:-false}"

# If ONLY_LLMAIL, skip others
if [[ "$ONLY_LLMAIL" == "true" ]]; then
    SKIP_FUJITSU=true
    SKIP_AGENTDOJO=true
    SKIP_LLMAIL=false
fi

# Limits for testing (set to empty for full processing)
FUJITSU_LIMIT="${FUJITSU_LIMIT:-}"
AGENTDOJO_LIMIT="${AGENTDOJO_LIMIT:-}"
LLMAIL_LIMIT="${LLMAIL_LIMIT:-}"

echo "========================================"
echo "ETL_A: Raw to B1 Skeleton Traces"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Processing:"
echo "  Fujitsu B4: $([[ $SKIP_FUJITSU == 'true' ]] && echo 'SKIP' || echo 'YES')"
echo "  AgentDojo: $([[ $SKIP_AGENTDOJO == 'true' ]] && echo 'SKIP' || echo 'YES')"
echo "  LLMail-Inject: $([[ $SKIP_LLMAIL == 'true' ]] && echo 'SKIP' || echo 'YES')"
echo ""

# =============================================================================
# Fujitsu B4 (Orchestrator Tool-Flip)
# =============================================================================
if [[ "$SKIP_FUJITSU" != "true" ]]; then
    echo "========================================"
    echo "Processing Fujitsu B4..."
    echo "========================================"
    
    FUJITSU_INPUT="$REPO_DIR/data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl"
    FUJITSU_OUTPUT="$OUTPUT_DIR/fujitsu_b4_skeletons.jsonl"
    
    if [[ -f "$FUJITSU_INPUT" ]]; then
        LIMIT_ARG=""
        if [[ -n "$FUJITSU_LIMIT" ]]; then
            echo "  Using limit: $FUJITSU_LIMIT"
            # Note: Fujitsu B4 doesn't have a limit arg, process all
        fi
        
        echo "  Input: $FUJITSU_INPUT"
        echo "  Output: $FUJITSU_OUTPUT"
        
        python src/schemas/tools/ETL_A.py \
            --fujitsu-b4 "$FUJITSU_INPUT" \
            --output "$FUJITSU_OUTPUT" \
            --split train
        
        echo "  ✓ Fujitsu B4: $(wc -l < "$FUJITSU_OUTPUT") traces"
    else
        echo "  ✗ Input not found: $FUJITSU_INPUT"
    fi
    echo ""
fi

# =============================================================================
# AgentDojo (Multi-turn Agent Attacks)
# =============================================================================
if [[ "$SKIP_AGENTDOJO" != "true" ]]; then
    echo "========================================"
    echo "Processing AgentDojo..."
    echo "========================================"
    
    # Process multiple AgentDojo files and merge
    AGENTDOJO_DIR="$REPO_DIR/data/agent_dojo"
    AGENTDOJO_OUTPUT="$OUTPUT_DIR/agentdojo_complete.jsonl"
    
    # Clear output file
    > "$AGENTDOJO_OUTPUT"
    
    for f in "$AGENTDOJO_DIR"/agentdojo-*.jsonl; do
        if [[ -f "$f" ]]; then
            basename=$(basename "$f")
            echo "  Processing: $basename"
            
            LIMIT_ARG=""
            if [[ -n "$AGENTDOJO_LIMIT" ]]; then
                LIMIT_ARG="--agentdojo-limit $AGENTDOJO_LIMIT"
            fi
            
            python src/schemas/tools/ETL_A.py \
                --agentdojo "$f" \
                $LIMIT_ARG \
                --output "/tmp/agentdojo_temp.jsonl" \
                --split train
            
            cat "/tmp/agentdojo_temp.jsonl" >> "$AGENTDOJO_OUTPUT"
            rm -f "/tmp/agentdojo_temp.jsonl"
        fi
    done
    
    if [[ -f "$AGENTDOJO_OUTPUT" ]]; then
        echo "  ✓ AgentDojo: $(wc -l < "$AGENTDOJO_OUTPUT") traces"
    fi
    echo ""
fi

# =============================================================================
# LLMail-Inject (Email Injection Attacks)
# =============================================================================
if [[ "$SKIP_LLMAIL" != "true" ]]; then
    echo "========================================"
    echo "Processing LLMail-Inject..."
    echo "========================================"
    
    LLMAIL_PHASE1="$REPO_DIR/data/llmail_inject/raw_submissions_phase1.jsonl"
    LLMAIL_PHASE2="$REPO_DIR/data/llmail_inject/raw_submissions_phase2.jsonl"
    LLMAIL_OUTPUT="$OUTPUT_DIR/llmail_inject_skeletons.jsonl"
    
    # Clear output
    > "$LLMAIL_OUTPUT"
    
    # Process Phase 1
    if [[ -f "$LLMAIL_PHASE1" ]]; then
        echo "  Processing Phase 1..."
        echo "  Input: $LLMAIL_PHASE1"
        
        LIMIT_ARG=""
        if [[ -n "$LLMAIL_LIMIT" ]]; then
            LIMIT_ARG="--llmail-inject-limit $LLMAIL_LIMIT"
            echo "  Using limit: $LLMAIL_LIMIT"
        fi
        
        python src/schemas/tools/ETL_A.py \
            --llmail-inject "$LLMAIL_PHASE1" \
            $LIMIT_ARG \
            --output "/tmp/llmail_phase1.jsonl" \
            --split train
        
        cat "/tmp/llmail_phase1.jsonl" >> "$LLMAIL_OUTPUT"
        rm -f "/tmp/llmail_phase1.jsonl"
        echo "  ✓ Phase 1 processed"
    else
        echo "  ✗ Phase 1 not found: $LLMAIL_PHASE1"
    fi
    
    # Process Phase 2
    if [[ -f "$LLMAIL_PHASE2" ]]; then
        echo "  Processing Phase 2..."
        echo "  Input: $LLMAIL_PHASE2"
        
        LIMIT_ARG=""
        if [[ -n "$LLMAIL_LIMIT" ]]; then
            LIMIT_ARG="--llmail-inject-limit $LLMAIL_LIMIT"
        fi
        
        python src/schemas/tools/ETL_A.py \
            --llmail-inject "$LLMAIL_PHASE2" \
            $LIMIT_ARG \
            --output "/tmp/llmail_phase2.jsonl" \
            --split train
        
        cat "/tmp/llmail_phase2.jsonl" >> "$LLMAIL_OUTPUT"
        rm -f "/tmp/llmail_phase2.jsonl"
        echo "  ✓ Phase 2 processed"
    else
        echo "  ✗ Phase 2 not found: $LLMAIL_PHASE2"
    fi
    
    if [[ -f "$LLMAIL_OUTPUT" ]]; then
        echo "  ✓ LLMail-Inject total: $(wc -l < "$LLMAIL_OUTPUT") traces"
    fi
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "========================================"
echo "ETL_A Complete"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
for f in "$OUTPUT_DIR"/*_skeletons.jsonl "$OUTPUT_DIR"/*_complete.jsonl; do
    if [[ -f "$f" ]]; then
        echo "  $(basename "$f"): $(wc -l < "$f") traces"
    fi
done
echo ""
echo "Next step: Run generate_completions.py to convert B1 -> B2"
echo "  sbatch slurm/pipeline/02_fill_skeletons.sbatch"
echo ""
echo "Done at $(date)"
