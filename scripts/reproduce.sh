#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ $# -eq 0 || "$1" == "--help" ]]; then
  echo 'Usage: bash scripts/reproduce.sh --check | --run --allow-api-cost'
  echo 'Prepare Python environments, frozen data, model, ALFWorld assets and .env first.'
  exit 0
fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ "$1" == --check && $# -eq 1 ]]; then
  exec "${SFT_PYTHON:-python}" src/tools/check.py
fi
if [[ $# -ne 2 || "$1" != --run || "$2" != --allow-api-cost ]]; then
  echo 'Full reproduction requires --run --allow-api-cost.' >&2; exit 2
fi
: "${CUDA_VISIBLE_DEVICES:?Explicitly set your allocated GPU}"
: "${ALFWORLD_DATA:?Set the ALFWorld asset directory}"
: "${RETHINK_API_KEY:?Provide your own hint API key}"
: "${RETHINK_API_ENDPOINT:?Provide your own hint API endpoint}"
"${SFT_PYTHON:-python}" src/tools/check.py --assets
bash scripts/train.sh sft
bash scripts/train.sh opsd --allow-api-cost
bash scripts/train.sh baseline
for ARM in opsd baseline; do
  EXPORT="$ROOT/checkpoints/${ARM}_hf150"
  PYTHONPATH="$ROOT/.deps/sdar" "${OPSD_PYTHON:-python}" \
    .deps/sdar/scripts/model_merger.py merge --backend fsdp \
    --local_dir "$ROOT/checkpoints/$ARM/checkpoints/global_step_150/actor" \
    --target_dir "$EXPORT"
  bash scripts/eval.sh --model "$EXPORT" --out "$ROOT/runs/${ARM}_seen140" --step 150
done
"${SFT_PYTHON:-python}" src/tools/plot.py \
  --baseline "$ROOT/runs/baseline_seen140/output/summary.json" \
  --method "$ROOT/runs/opsd_seen140/output/summary.json"

