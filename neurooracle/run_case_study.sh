#!/usr/bin/env bash
# Autoresearch cycle for a registered Nature-paper case study.
# Usage: bash neurooracle/run_case_study.sh <case_study_name> [run_id] [stages]
#   case_study_name : case1_transdiagnostic | case2_pathway_mediation | case3_hindcasting
#   run_id          : optional sub-directory tag (default: 001)
#   stages          : comma list, default "batch,novelty,critic,plausibility"
set -euo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if [ -f .env.keys ]; then
    set -a; source .env.keys; set +a
fi

CASE_NAME="${1:?case study name required (case1_transdiagnostic | case2_pathway_mediation | case3_hindcasting)}"
RUN_ID="${2:-001}"
STAGES="${3:-batch,novelty,critic,plausibility}"

PY=/c/Users/45846/anaconda3/envs/neuroclaw/python.exe
KG=neurooracle/data/full_snapshot_v2/knowledge_graph.json
KGE=neurooracle/data/full_snapshot_v2/kge_complex.pt
NOV_CACHE=neurooracle/data/full_snapshot_v2/novelty_cache.json
OUT_DIR="neurooracle/data/cs_runs/${CASE_NAME}/${RUN_ID}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

# Reuse upstream novelty cache so we don't re-hit PubMed
cp -n "$NOV_CACHE" "$OUT_DIR/novelty_cache.json" 2>/dev/null || true

echo "=== Case study '$CASE_NAME' run '$RUN_ID' started at $(date) ===" | tee -a "$LOG"
echo "    stages: $STAGES" | tee -a "$LOG"

"$PY" -m neurooracle.src.hypothesis_cli --graph "$KG" case-study "$CASE_NAME" \
    --output-dir "$OUT_DIR" \
    --stages "$STAGES" \
    --kge "$KGE" \
    --kg-for-plausibility "$KG" \
    2>&1 | tee -a "$LOG"

echo "=== Case study '$CASE_NAME' run '$RUN_ID' done at $(date) ===" | tee -a "$LOG"
