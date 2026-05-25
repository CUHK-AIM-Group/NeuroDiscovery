#!/usr/bin/env bash
# Hypothesis generation + validation cycle.
# Usage: bash neurooracle/run_cycle.sh <cycle_id> [seed_from_path]
#   seed_from_path: optional prior cycle's hypotheses_final.json -> evolve mode
set -euo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if [ -f .env.keys ]; then
    set -a; source .env.keys; set +a
fi

CYCLE_ID="${1:-001}"
SEED_FROM="${2:-}"
PY=/c/Users/45846/anaconda3/envs/neuroclaw/python.exe
KG=neurooracle/data/full/knowledge_graph.json
KGE=neurooracle/data/full/kge_complex.pt
NOV_CACHE=neurooracle/data/full/novelty_cache.json
OUT_DIR="neurooracle/data/cycles/cycle_${CYCLE_ID}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/cycle.log"

# Reuse upstream novelty cache so we don't re-hit PubMed
cp -n "$NOV_CACHE" "$OUT_DIR/novelty_cache.json" 2>/dev/null || true

echo "=== Cycle $CYCLE_ID started at $(date) ===" | tee -a "$LOG"

if [ -n "$SEED_FROM" ]; then
    echo "[1/4] evolve from seeds: $SEED_FROM" | tee -a "$LOG"
    cp "$SEED_FROM" "$OUT_DIR/hypotheses_seeds.json"
    "$PY" -m neurooracle.src.hypothesis_cli --graph "$KG" evolve \
        --input "$OUT_DIR/hypotheses_seeds.json" \
        --output "$OUT_DIR/hypotheses_raw.json" \
        --population 50 --generations 5 \
        --mutation-rate 0.6 --crossover-rate 0.3 \
        --tournament-size 3 --elitism 5 \
        2>&1 | tee -a "$LOG"
else
    echo "[1/4] batch generate" | tee -a "$LOG"
    "$PY" -m neurooracle.src.hypothesis_cli --graph "$KG" batch \
        --output "$OUT_DIR/hypotheses_raw.json" \
        --max-hops 3 --max-paths 4 --max-seeds 30 \
        --tasks biomarker_discovery,brain_age,disease_subtyping,progression_prediction \
        --target-per-task 100 --max-retries 4 --retry-scale 2.0 \
        2>&1 | tee -a "$LOG"
fi

# Stage 2: novelty against PubMed + Semantic Scholar (top 200)
echo "[2/4] novelty check" | tee -a "$LOG"
"$PY" -m neurooracle.src.hypothesis_cli --graph "$KG" novelty \
    --input "$OUT_DIR/hypotheses_raw.json" \
    --output "$OUT_DIR/hypotheses_novel.json" \
    --top 200 --alpha 0.5 \
    2>&1 | tee -a "$LOG"

# Stage 3: critic agent (top 50)
echo "[3/4] critic refinement" | tee -a "$LOG"
"$PY" -m neurooracle.src.hypothesis_cli --graph "$KG" critic \
    --input "$OUT_DIR/hypotheses_novel.json" \
    --output "$OUT_DIR/hypotheses_critic.json" \
    --top 100 --max-rounds 2 --threshold 0.55 --max-workers 12 \
    2>&1 | tee -a "$LOG"

# Stage 4: KGE plausibility scoring
echo "[4/4] plausibility" | tee -a "$LOG"
"$PY" -m neurooracle.src.hypothesis_cli plausibility \
    --input "$OUT_DIR/hypotheses_critic.json" \
    --kge "$KGE" \
    --kg "$KG" \
    --output "$OUT_DIR/hypotheses_final.json" \
    --novelty-cache "$OUT_DIR/novelty_cache.json" \
    --top 100 \
    2>&1 | tee -a "$LOG"

echo "=== Cycle $CYCLE_ID done at $(date) ===" | tee -a "$LOG"
