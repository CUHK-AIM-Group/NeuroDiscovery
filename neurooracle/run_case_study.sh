#!/usr/bin/env bash
# Autoresearch cycle for a registered Nature-paper case study.
# Usage: bash neurooracle/run_case_study.sh <case_study_name> [run_id] [stages]
#   case_study_name : any ID reported by `hypothesis_cli case-study --help`
#   run_id          : optional sub-directory tag (default: 001)
#   stages          : comma list, default "batch,novelty,critic,plausibility"
set -euo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if [ -f .env.keys ]; then
    set -a; source .env.keys; set +a
fi

CASE_NAME="${1:?registered case study ID required}"
RUN_ID="${2:-001}"
STAGES="${3:-batch,novelty,critic,plausibility}"

PY="${PYTHON:-python}"
KG_ARGS=()
if [ -n "${NEUROCLAW_GRAPH_PATH:-}" ]; then KG_ARGS=(--graph "$NEUROCLAW_GRAPH_PATH"); fi
KG=$("$PY" -m neurooracle.current_graph "${KG_ARGS[@]}")
KGE_ARGS=()
if [ -n "${NEUROCLAW_KGE_CHECKPOINT:-}" ]; then KGE_ARGS=(--kge "$NEUROCLAW_KGE_CHECKPOINT"); fi
if [[ ",${STAGES// /}," == *,plausibility,* ]] && [ "${#KGE_ARGS[@]}" -eq 0 ]; then
    echo "Set NEUROCLAW_KGE_CHECKPOINT to a checkpoint matched to the selected graph." >&2
    exit 1
fi
NOV_CACHE="${NEUROCLAW_NOVELTY_CACHE:-neurooracle/data/cache/novelty_cache.json}"
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
    "${KGE_ARGS[@]}" \
    --kg-for-plausibility "$KG" \
    2>&1 | tee -a "$LOG"

echo "=== Case study '$CASE_NAME' run '$RUN_ID' done at $(date) ===" | tee -a "$LOG"
