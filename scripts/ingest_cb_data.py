#!/usr/bin/env python3
"""
Circuit Breakers (Representation Rerouting) Data Ingestion Script

This script ingests data to create a dataset for Circuit Breaker training (Representation Rerouting).
It pairs harmful prompts (attacks) with benign capability tasks to train models to refuse the former 
while maintaining performance on the latter.

TRAINING METHODOLOGY (per Gray-Swan Circuit Breakers paper):
================================================================================
The key insight is that batches must be BALANCED (1:1 harmful:benign) at every step.
The optimization objective requires both inputs simultaneously:
  - Rerouting Loss (from harmful data): Push representations away from harmful vector
  - Retain Loss (from benign data): Anchor representations to benign vector

DO NOT imbalance the data ratio. The balance is controlled by the ALPHA SCHEDULE (λ),
not by the raw data count:
  - Early training: λ is high (e.g., 1.5 for Llama-3) → heavy weight on harmful
  - Late training: λ decays linearly to 0 → transitions to benign-focused

AGENT DATA MIXING RECOMMENDATIONS:
  - Harmful batch (D_rr): 50% Agent attacks + 50% General LLM attacks
  - Benign batch (D_ret): 50% Agent tool-use + 50% General chat/refusal
================================================================================

It generates:
1. Harmful Pairs: JSONL files containing attack prompts and metadata.
   Output: data/circuit_breakers/harmful/harmful_pairs.jsonl
2. Benign Pairs: JSONL files containing capability tasks and expected behaviors.
   Output: data/circuit_breakers/benign/benign_pairs.jsonl
3. Training Batches: JSONL file with STRICTLY balanced batches (1:1 harmful:benign).
   Output: data/circuit_breakers/cb_training_batches.jsonl

DATA DEPENDENCIES:
This script relies on the following local files.

1. Fujitsu Orchestrator Attacks (Harmful)
   - Path: data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl
   - Description: Large collection of indirect prompt injection attacks.
   - Usage: Source of harmful attack prompts.

2. AgentDojo (Harmful & Benign)
   - Path: data/agent_dojo/agentdojo-claude-3-5-sonnet-20241022.jsonl
   - Description: Execution traces.
   - Usage: 
     - Harmful: Traces where metadata.security is False (successful attacks).
     - Benign: Traces where metadata.security is True or success is True.

3. WebArena (Benign)
   - Path: data/webarena/config_files/test.raw.json
   - Description: Task definitions.
   - Usage: Source of benign capability prompts (intents).

4. TAU2 (Benign)
   - Path: data/tau2_repo/data/tau2/domains/{telecom,airline,retail,mock}/tasks.json
   - Description: Task definitions.
   - Usage: Source of benign capability prompts (instructions).

5. WebLINX (Benign)
   - Path: data/processed/weblinx_sample.json
   - Description: Sample of WebLINX data.
   - Usage: Source of benign capability prompts (utterances).

OUTPUT FORMAT (Pairs):
{
    "id": "unique_id",
    "source": "dataset_name",
    "category": "attack_type_or_capability",
    "prompt": "The text prompt",
    "metadata": { ... }
}
"""

import os
import json
import random
import re
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠️  Pandas not found. AttackQA (parquet) will be skipped.")

# --- Configuration & Paths ---
if Path.cwd().name == 'scripts':
    BASE_DIR = Path('..').resolve()
else:
    BASE_DIR = Path('.').resolve()

DATA_DIR = BASE_DIR / 'data'
CB_DIR = DATA_DIR / 'circuit_breakers'

# Define expected input paths
PATH_FUJITSU = DATA_DIR / 'fujitsu' / 'orchestrator_attacks_combined_deduplicated.jsonl'
PATH_AGENTDOJO_DIR = DATA_DIR / 'agent_dojo'
PATH_AGENTDOJO_DEFAULT = PATH_AGENTDOJO_DIR / 'agentdojo-claude-3-5-sonnet-20241022.jsonl'
PATH_WEBARENA = DATA_DIR / 'webarena' / 'config_files' / 'test.raw.json'
PATH_TAU2_REPO = DATA_DIR / 'tau2_repo' / 'data' / 'tau2' / 'domains'
PATH_WEBLINX = DATA_DIR / 'processed' / 'weblinx_sample.json'
PATH_ATTACKQA = DATA_DIR / 'attackqa' / 'attackqa.parquet'
PATH_AGENTHARM_TEST = DATA_DIR / 'agent_harm' / 'harmful_behaviors_test_public.json'
PATH_AGENTHARM_VAL = DATA_DIR / 'agent_harm' / 'harmful_behaviors_validation.json'

# Additional Fujitsu benchmarks (see DATA.md)
PATH_FUJITSU_RAG_POISONING = DATA_DIR / 'fujitsu' / 'rag_poisoning_benchmark_combined_deduplicated.jsonl'
PATH_FUJITSU_SAFETY_DIRECT = DATA_DIR / 'fujitsu' / 'safety_benchmark_direct_query_combined_deduplicated.jsonl'
PATH_FUJITSU_IMAGE_POISONING = DATA_DIR / 'fujitsu' / 'image_poisoning_simulation_results_20250504_202954.jsonl'

# --- Utility Functions ---

def setup_directories():
    """Ensure output directories exist."""
    (CB_DIR / 'harmful').mkdir(parents=True, exist_ok=True)
    (CB_DIR / 'benign').mkdir(parents=True, exist_ok=True)
    print(f"Directories ready: {CB_DIR}")

def read_json(path: Path) -> Any:
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON {path}: {e}")
        return None

def read_jsonl(path: Path, limit: int = None) -> List[Dict[str, Any]]:
    items = []
    if not path.exists():
        print(f"⚠️  File not found: {path}")
        return items
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"❌ Error reading JSONL {path}: {e}")
    return items

def list_agentdojo_jsonl_files() -> List[Path]:
    """Return AgentDojo trace files to ingest.

    By default, ingest *all* JSONL files under data/agent_dojo matching agentdojo-*.jsonl.
    Falls back to the historical default file if no matches are found.
    """
    files: List[Path] = []
    if PATH_AGENTDOJO_DIR.exists():
        files = sorted([p for p in PATH_AGENTDOJO_DIR.glob('agentdojo-*.jsonl') if p.is_file()])
    if not files and PATH_AGENTDOJO_DEFAULT.exists():
        files = [PATH_AGENTDOJO_DEFAULT]
    return files

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        print(f"⚠️  No data to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"✅ Wrote {len(rows)} rows to {path}")
    except Exception as e:
        print(f"❌ Error writing to {path}: {e}")

def backup_if_exists(path: Path, backup_dir: Path):
    """Create a timestamped backup copy of a file if it already exists."""
    try:
        if not path.exists() or not path.is_file():
            return
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"{path.name}.{ts}.bak"
        shutil.copy2(path, backup_path)
        print(f"🗂️  Backed up existing {path.name} → {backup_path}")
    except Exception as e:
        print(f"⚠️  Backup failed for {path}: {e}")

def norm_text(x: Any) -> str:
    """Normalize text by stripping whitespace and collapsing multiple spaces."""
    s = str(x or '').strip()
    s = re.sub(r"\s+", " ", s)
    return s

# --- Data Loading & Processing ---

def load_harmful_data() -> List[Dict[str, Any]]:
    """Load and normalize harmful data from Fujitsu, AgentDojo, and AgentHarm."""
    harmful_pairs = []
    counts = {'fujitsu_b4_orchestrator': 0, 'fujitsu_b1_rag_poisoning': 0, 'fujitsu_b3_direct_query': 0, 'fujitsu_b2_image_poisoning': 0, 'agentdojo': 0, 'agentharm': 0}
    
    # 1. Fujitsu (B4) Orchestrator Attacks
    print("\n--- Loading Harmful: Fujitsu (B4 Orchestrator) ---")
    fujitsu_data = read_jsonl(PATH_FUJITSU)  # No limit
    for a in fujitsu_data:
        prompt = norm_text(a.get('malicious_injection'))
        if not prompt:
            continue

        harmful_pairs.append({
            'id': f"fujitsu_b4_{a.get('attack_id', a.get('record_id', 'na'))}",
            'source': 'fujitsu',
            'category': a.get('category', 'unknown'),
            'subtype': a.get('subtype', 'unknown'),
            'attack_prompt': prompt,
            'benign_query': norm_text(a.get('benign_query')),
            'metadata': {
                'benchmark': 'B4_orchestrator',
                'combined_query': a.get('combined_query'),
                'expected_tool': a.get('expected_tool'),
                'simulated_tool': a.get('simulated_tool'),
                'success': a.get('success'),
                'judge_note': a.get('judge_note'),
                'record_id': a.get('record_id'),
            }
        })
        counts['fujitsu_b4_orchestrator'] += 1
    print(f"  Loaded: {counts['fujitsu_b4_orchestrator']}")

    # 1b. Fujitsu (B1) RAG Poisoning Benchmark
    print("\n--- Loading Harmful: Fujitsu (B1 RAG Poisoning) ---")
    b1 = read_jsonl(PATH_FUJITSU_RAG_POISONING)
    for a in b1:
        poison = norm_text(a.get('poison_content'))
        if not poison:
            continue
        harmful_pairs.append({
            'id': f"fujitsu_b1_{a.get('id', 'na')}",
            'source': 'fujitsu',
            'category': a.get('attack_category', 'LLM04_Data_Poisoning'),
            'subtype': a.get('attack_subtype', a.get('attack_mechanism', 'unknown')),
            'attack_prompt': poison,
            'benign_query': norm_text(a.get('user_query')),
            'metadata': {
                'benchmark': 'B1_rag_poisoning',
                'adversarial_goal': a.get('adversarial_goal'),
                'attack_mechanism': a.get('attack_mechanism'),
                'target_rag_component': a.get('target_rag_component'),
                'target_component_model': a.get('target_component_model'),
                'judge_assessment': a.get('judge_assessment'),
            }
        })
        counts['fujitsu_b1_rag_poisoning'] += 1
    print(f"  Loaded: {counts['fujitsu_b1_rag_poisoning']}")

    # 1c. Fujitsu (B3) Safety Benchmark Direct Query
    print("\n--- Loading Harmful: Fujitsu (B3 Direct Query) ---")
    b3 = read_jsonl(PATH_FUJITSU_SAFETY_DIRECT)
    for a in b3:
        user_query = norm_text(a.get('user_query'))
        if not user_query:
            continue
        harmful_pairs.append({
            'id': f"fujitsu_b3_{a.get('id', 'na')}",
            'source': 'fujitsu',
            'category': a.get('attack_category', 'unknown'),
            'subtype': a.get('attack_subtype', 'unknown'),
            'attack_prompt': user_query,
            'benign_query': '',
            'metadata': {
                'benchmark': 'B3_direct_query',
                'adversarial_goal': a.get('adversarial_goal'),
                'judge_assessment': a.get('judge_assessment'),
                'mode': a.get('mode'),
                'manipulation_strategy': a.get('manipulation_strategy'),
                'intent_opacity': a.get('intent_Opacity'),
                'obfuscation': a.get('Obfuscation'),
            }
        })
        counts['fujitsu_b3_direct_query'] += 1
    print(f"  Loaded: {counts['fujitsu_b3_direct_query']}")

    # 1d. Fujitsu (B2) Image Poisoning Simulation Results
    print("\n--- Loading Harmful: Fujitsu (B2 Image Poisoning) ---")
    b2 = read_jsonl(PATH_FUJITSU_IMAGE_POISONING)
    for a in b2:
        payload = norm_text(a.get('poison_payload'))
        if not payload:
            continue
        harmful_pairs.append({
            'id': f"fujitsu_b2_{a.get('attack_id', 'na')}",
            'source': 'fujitsu',
            'category': 'image_poisoning',
            'subtype': a.get('attack_subtype', 'unknown'),
            'attack_prompt': payload,
            'benign_query': norm_text(a.get('user_query')),
            'metadata': {
                'benchmark': 'B2_image_poisoning',
                'image_type': a.get('image_type'),
                'adversarial_goal': a.get('adversarial_goal'),
                'baseline_judge_assessment': a.get('baseline_judge_assessment'),
                'mta_judge_assessment': a.get('mta_judge_assessment'),
            }
        })
        counts['fujitsu_b2_image_poisoning'] += 1
    print(f"  Loaded: {counts['fujitsu_b2_image_poisoning']}")
    
    # 2. AgentDojo Harmful (Security Violations)
    print("\n--- Loading Harmful: AgentDojo (all traces in data/agent_dojo) ---")
    agentdojo_files = list_agentdojo_jsonl_files()
    if not agentdojo_files:
        print("  ⚠️  No AgentDojo JSONL files found.")
    for p in agentdojo_files:
        agentdojo_data = read_jsonl(p)  # No limit
        stem = p.stem
        for i, r in enumerate(agentdojo_data):
            md = r.get('metadata', {})
            if md.get('security') is False:  # Explicit security failure = Harmful
                msgs = r.get('messages', [])
                user_msg = next((m.get('content') for m in msgs if m.get('role') == 'user'), '')
                prompt = norm_text(user_msg)
                if not prompt:
                    continue

                harmful_pairs.append({
                    'id': f"agentdojo_harm_{stem}_{md.get('user_task_id', i)}",
                    'source': 'agentdojo',
                    'category': md.get('suite_name', 'agentdojo'),
                    'subtype': 'security_violation',
                    'attack_prompt': prompt,
                    'benign_query': '',
                    'metadata': {
                        **md,
                        'origin_file': p.name,
                    }
                })
                counts['agentdojo'] += 1
    print(f"  Loaded: {counts['agentdojo']}")
    
    # 3. AgentHarm (Harmful behavior prompts)
    print("\n--- Loading Harmful: AgentHarm ---")
    behaviors = []
    for p in [PATH_AGENTHARM_TEST, PATH_AGENTHARM_VAL]:
        d = read_json(p)
        if d:
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, list): behaviors.extend(v)
                    elif isinstance(v, dict): behaviors.append(v)
            elif isinstance(d, list):
                behaviors.extend(d)
    
    for i, b in enumerate(behaviors):
        prompt = norm_text(b.get('prompt') or b.get('behavior') or b.get('Behavior'))
        if not prompt:
            continue
        harmful_pairs.append({
            'id': f"agentharm_{i:04d}",
            'source': 'agentharm',
            'category': b.get('FunctionalCategory', b.get('category', 'harmful')),
            'subtype': 'harmful_behavior',
            'attack_prompt': prompt,
            'benign_query': '',
            'metadata': {}
        })
        counts['agentharm'] += 1
    print(f"  Loaded: {counts['agentharm']}")
    
    print(
        "\n  HARMFUL TOTALS: "
        f"Fujitsu(B4)={counts['fujitsu_b4_orchestrator']}, "
        f"Fujitsu(B1)={counts['fujitsu_b1_rag_poisoning']}, "
        f"Fujitsu(B3)={counts['fujitsu_b3_direct_query']}, "
        f"Fujitsu(B2)={counts['fujitsu_b2_image_poisoning']}, "
        f"AgentDojo={counts['agentdojo']}, "
        f"AgentHarm={counts['agentharm']}"
    )
    return harmful_pairs

def load_benign_data() -> List[Dict[str, Any]]:
    """Load and normalize benign data from WebArena, TAU2, WebLINX, AgentDojo, AttackQA."""
    benign_pairs = []
    counts = {'webarena': 0, 'tau2': 0, 'weblinx': 0, 'agentdojo': 0, 'attackqa': 0}
    
    # 1. WebArena
    print("\n--- Loading Benign: WebArena ---")
    wa_tasks = read_json(PATH_WEBARENA) or []
    for t in wa_tasks:
        prompt = norm_text(t.get('intent'))
        if not prompt:
            continue
        benign_pairs.append({
            'id': f"webarena_{t.get('task_id')}",
            'source': 'webarena',
            'category': 'capability',
            'prompt': prompt,
            'metadata': {
                'start_url': t.get('start_url'),
                'eval_types': t.get('eval', {}).get('eval_types')
            }
        })
        counts['webarena'] += 1
    print(f"  Loaded: {counts['webarena']}")

    # 2. TAU2
    print("\n--- Loading Benign: TAU2 ---")
    if PATH_TAU2_REPO.exists():
        for dom in ['telecom', 'airline', 'retail', 'mock']:
            p = PATH_TAU2_REPO / dom / 'tasks.json'
            d = read_json(p) or []
            for i, t in enumerate(d):
                # TAU2 has nested structure: user_scenario.instructions.task_instructions
                prompt = None
                try:
                    us = t.get('user_scenario')
                    if us and isinstance(us, dict):
                        instr = us.get('instructions')
                        if instr and isinstance(instr, dict):
                            prompt = norm_text(instr.get('task_instructions'))
                except Exception:
                    pass
                if not prompt:
                    prompt = norm_text(t.get('instruction') or t.get('title') or t.get('id'))
                if not prompt:
                    continue
                benign_pairs.append({
                    'id': f"tau2_{dom}_{i:04d}",
                    'source': 'tau2',
                    'category': 'capability',
                    'prompt': prompt,
                    'metadata': {'domain': dom}
                })
                counts['tau2'] += 1
    print(f"  Loaded: {counts['tau2']}")

    # 3. WebLINX
    print("\n--- Loading Benign: WebLINX ---")
    wl_data = read_json(PATH_WEBLINX) or []
    for i, s in enumerate(wl_data):
        prompt = norm_text(s.get('utterances'))
        if not prompt:
            continue
        benign_pairs.append({
            'id': f"weblinx_{i:06d}",
            'source': 'weblinx',
            'category': 'capability',
            'prompt': prompt,
            'metadata': {'clean_html_len': len(str(s.get('clean_html', '')))}
        })
        counts['weblinx'] += 1
    print(f"  Loaded: {counts['weblinx']}")

    # 4. AgentDojo Benign
    print("\n--- Loading Benign: AgentDojo (all traces in data/agent_dojo) ---")
    agentdojo_files = list_agentdojo_jsonl_files()
    if not agentdojo_files:
        print("  ⚠️  No AgentDojo JSONL files found.")
    for p in agentdojo_files:
        ad_data = read_jsonl(p)  # No limit
        stem = p.stem
        for i, r in enumerate(ad_data):
            md = r.get('metadata', {})
            if md.get('security') is True or md.get('success') is True:
                msgs = r.get('messages', [])
                user_msg = next((m.get('content') for m in msgs if m.get('role') == 'user'), '')
                prompt = norm_text(user_msg)
                if not prompt:
                    continue
                benign_pairs.append({
                    'id': f"agentdojo_benign_{stem}_{i:06d}",
                    'source': 'agentdojo',
                    'category': 'capability',
                    'prompt': prompt,
                    'metadata': {
                        **md,
                        'origin_file': p.name,
                    }
                })
                counts['agentdojo'] += 1
    print(f"  Loaded: {counts['agentdojo']}")
    
    # 5. AttackQA (Security Knowledge QA - benign competency)
    print("\n--- Loading Benign: AttackQA ---")
    if pd is not None and PATH_ATTACKQA.exists():
        try:
            df = pd.read_parquet(PATH_ATTACKQA)
            for i, row in df.iterrows():
                question = str(row.get('question', '')).strip()
                if not question:
                    continue
                benign_pairs.append({
                    'id': f"attackqa_{i:06d}",
                    'source': 'attackqa',
                    'category': 'competency',
                    'prompt': question,
                    'metadata': {'answer': str(row.get('answer', ''))[:200]}
                })
                counts['attackqa'] += 1
        except Exception as e:
            print(f"  ❌ Error reading parquet: {e}")
    else:
        print("  ⚠️  Skipped (pandas or file missing).")
    print(f"  Loaded: {counts['attackqa']}")

    print(f"\n  BENIGN TOTALS: WebArena={counts['webarena']}, TAU2={counts['tau2']}, WebLINX={counts['weblinx']}, AgentDojo={counts['agentdojo']}, AttackQA={counts['attackqa']}")
    return benign_pairs

def create_batches(harmful: List[Dict], benign: List[Dict], batch_size: int = 16) -> List[Dict]:
    """
    Create STRICTLY balanced batches (1:1 harmful:benign).
    
    Per Circuit Breaker methodology (Algorithm 1):
    - Every batch MUST have equal harmful and benign samples
    - This is NON-NEGOTIABLE: the loss function requires both D_rr and D_ret
    - The alpha schedule controls the loss weight, NOT the data ratio
    
    Args:
        harmful: List of harmful/attack samples (D_rr)
        benign: List of benign/capability samples (D_ret)
        batch_size: Total batch size (will be split 50/50)
    
    Returns:
        List of batches, each with batch_size/2 harmful + batch_size/2 benign
    """
    random.shuffle(harmful)
    random.shuffle(benign)
    
    # STRICT 1:1 ratio - exactly half harmful, half benign
    samples_per_side = batch_size // 2  # 8 harmful + 8 benign for batch_size=16
    
    batches = []
    h_idx, b_idx = 0, 0
    
    # Limit by the smaller dataset to ensure perfect 1:1 balance
    max_batches = min(len(harmful), len(benign)) // samples_per_side
    
    while h_idx + samples_per_side <= len(harmful) and b_idx + samples_per_side <= len(benign):
        batch = {
            'harmful': harmful[h_idx : h_idx + samples_per_side],
            'benign': benign[b_idx : b_idx + samples_per_side]
        }
        h_idx += samples_per_side
        b_idx += samples_per_side
        batches.append(batch)
        
    return batches

# --- Main Execution ---

def main():
    setup_directories()

    harmful_pairs = load_harmful_data()
    benign_pairs = load_benign_data()

    print("\n" + "="*60)
    print("              CIRCUIT BREAKER INGESTION SUMMARY")
    print("="*60)
    print(f"  Total Harmful Pairs: {len(harmful_pairs)}")
    print(f"  Total Benign Pairs:  {len(benign_pairs)}")

    # Identify limiting reagent
    if harmful_pairs and benign_pairs:
        ratio = len(harmful_pairs) / len(benign_pairs)
        if ratio > 2:
            print(f"\n  ⚠️  LIMITING REAGENT: BENIGN data")
            print(f"      You have {len(harmful_pairs)} harmful but only {len(benign_pairs)} benign.")
            print(f"      Ratio: {ratio:.1f}:1 (harmful:benign). Consider adding more benign data.")
        elif ratio < 0.5:
            print(f"\n  ⚠️  LIMITING REAGENT: HARMFUL data")
            print(f"      You have {len(benign_pairs)} benign but only {len(harmful_pairs)} harmful.")
            print(f"      Ratio: 1:{1/ratio:.1f} (harmful:benign). Consider adding more harmful data.")
        else:
            print(f"\n  ✅ Data is reasonably balanced (ratio: {ratio:.2f}:1 harmful:benign).")
    print("="*60)

    # Write raw pairs (Step 1 of pipeline)
    backups_dir = CB_DIR / '_backups'
    harmful_out = CB_DIR / 'harmful' / 'harmful_pairs.jsonl'
    benign_out = CB_DIR / 'benign' / 'benign_pairs.jsonl'

    backup_if_exists(harmful_out, backups_dir)
    backup_if_exists(benign_out, backups_dir)

    write_jsonl(harmful_out, harmful_pairs)
    write_jsonl(benign_out, benign_pairs)

    print("\n" + "="*60)
    print("  NEXT STEPS:")
    print("="*60)
    print("  1. Extract completions (REQUIRED for proper CB training):")
    print("     python scripts/format_for_cb/split_out_cb_completions.py --fujitsu-success-only")
    print("")
    print("  2. Create training batches from completions:")
    print("     python scripts/format_for_cb/create_cb_batches.py")
    print("")
    print("  Or run the full pipeline:")
    print("     python scripts/prepare_cb_training.py")
    print("="*60)

if __name__ == "__main__":
    main()
