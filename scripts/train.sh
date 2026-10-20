#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ $# -eq 0 || "$1" == "--help" ]]; then
  echo 'Usage: bash scripts/train.sh {sft|opsd|baseline} [options]'
  echo 'OPSD requires --allow-api-cost. SFT forwards resume/preflight options.'
  exit 0
fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
STAGE=$1; shift
: "${CUDA_VISIBLE_DEVICES:?Explicitly set your allocated GPU}"
export TOKENIZERS_PARALLELISM=false
if [[ "$STAGE" == sft ]]; then
  PY=${SFT_PYTHON:-python}
  export PYTHONPATH="$ROOT/src/sft"
  export HF_HUB_OFFLINE=1
  "$PY" -m route_bce.preflight --data "$ROOT/data/route_sft" \
    --tokenizer "$ROOT/checkpoints/Qwen2.5-1.5B-Instruct" \
    --output "$ROOT/reports/cpu_token_preflight_remote.json"
  export ALLOW_ROUTE_SEMANTIC_BCE_TRAINING=1
  exec "$PY" -m route_bce.train --root "$ROOT" --config "$ROOT/configs/sft.json" "$@"
elif [[ "$STAGE" == opsd || "$STAGE" == baseline ]]; then
  : "${ALFWORLD_DATA:?Set the ALFWorld asset directory}"
  PY=${OPSD_PYTHON:-python}
  if [[ ! -f .deps/sdar/.rethink_setup.json ]]; then
    echo 'Run python src/tools/setup.py --framework first.' >&2; exit 2
  fi
  RUN="$ROOT/checkpoints/$STAGE"
  MODEL="$ROOT/checkpoints/route_sft/global_step_2000"
  EXTRA=()
  if [[ "$STAGE" == baseline ]]; then
    MODEL="$ROOT/checkpoints/Qwen2.5-1.5B-Instruct"; EXTRA=(--baseline)
  fi
  if [[ ! -f "$RUN/setup.json" ]]; then
    "$PY" src/opsd/prepare.py --assets "$ALFWORLD_DATA" --model "$MODEL" \
      --run-dir "$RUN" "${EXTRA[@]}"
  fi
  exec "$PY" src/opsd/run.py --run-dir "$RUN" "$@"
else
  echo 'Unknown stage; expected sft, opsd or baseline.' >&2; exit 2
fi

