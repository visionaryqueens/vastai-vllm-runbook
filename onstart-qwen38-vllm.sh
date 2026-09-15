#!/bin/bash
# Qwen3.8-27B Uncensored (huihui abliterated) — vLLM FP8 + MTP, 256K ctx
# Sirve OpenAI-compatible API en :8000 con Bearer key.
# Probado en H200 NVL 143GB (~112 tok/s con MTP). Funciona en 80GB+ (H100, A100-80G) vía fp8.
set -uo pipefail

MODEL_REPO="${MODEL_REPO:-huihui-ai/Huihui-Qwen3.8-27B-abliterated}"
MODEL_DIR=/workspace/model
SERVED_NAME="${SERVED_NAME:-qwen38-uncensored}"
CTX="${CTX:-262144}"
GPU_UTIL="${GPU_UTIL:-0.92}"
# MAX_SEQS: el default de vLLM (1024) excede los Mamba cache blocks disponibles (~983) en esta
# arquitectura hibrida, y eso hace fallar el arranque con MTP -- no por incompatibilidad, sino por
# un off-by-41. Con 512 sobra margen para un solo usuario y MTP levanta (medido: +61% de decode).
MAX_SEQS="${MAX_SEQS:-512}"
PORT=8000

mkdir -p /workspace
exec > >(tee -a /workspace/onstart.log) 2>&1
status(){ echo "$*" > /workspace/STATUS; echo "=== [$(date -u +%H:%M:%S)] $*"; }

status "onstart arrancando (repo=$MODEL_REPO)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "WARN: nvidia-smi falló"

# VLLM_API_KEY: opcional. Si no la pasás, se genera una y se muestra en /workspace/STATUS.
: "${VLLM_API_KEY:=$(python3 -c 'import secrets; print(secrets.token_hex(16))')}"
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
export VLLM_API_KEY HF_HUB_ENABLE_HF_TRANSFER=1
status "VLLM_API_KEY=$VLLM_API_KEY  (Bearer token del endpoint :8000 — guardala)"

# ---------- 1. Descarga ----------
if [ ! -f "$MODEL_DIR/config.json" ]; then
  status "descargando $MODEL_REPO (BF16, ~56 GB) — puede tardar 5-20 min"
  pip install -q hf_transfer 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$MODEL_REPO", local_dir="$MODEL_DIR",
                  max_workers=16, ignore_patterns=["*.pth","*.msgpack","*.h5","original/*"])
print("download OK")
PY
fi
[ -f "$MODEL_DIR/config.json" ] || { status "FATAL: descarga falló, no hay config.json"; ls -la "$MODEL_DIR"; exit 1; }
status "modelo en disco: $(du -sh $MODEL_DIR | cut -f1)"

# ---------- 1b. Chat template: permitir >1 system message (clientes agenticos) ----------
CHAT_TEMPLATE=/workspace/chat_template.jinja
python3 - "$MODEL_DIR/tokenizer_config.json" "$MODEL_DIR/chat_template.jinja" "$CHAT_TEMPLATE" <<'PYT'
import json, sys, os
cfg, sep, out = sys.argv[1], sys.argv[2], sys.argv[3]
ct = None
try:
    ct = json.load(open(cfg)).get("chat_template")
except Exception:
    pass
if not ct and os.path.exists(sep):
    ct = open(sep).read()
if not ct:
    sys.exit(1)
ct = ct.replace("{{- raise_exception('System message must be at the beginning.') }}",
                "{#- system no-inicial permitido (override multi-system para clientes agenticos) #}")
open(out, "w").write(ct)
PYT
if [ -f "$CHAT_TEMPLATE" ]; then
  status "chat_template parcheado: $CHAT_TEMPLATE"
else
  status "WARN: no se pudo derivar chat_template — vLLM usará el embebido del modelo"
  unset CHAT_TEMPLATE
fi

# ---------- 2. Tool-parser compatible con esta build ----------
HELP=$(python3 -m vllm.entrypoints.openai.api_server --help 2>&1 || true)
PARSERS=""
for p in qwen3_coder qwen3_xml hermes; do grep -q -- "$p" <<<"$HELP" && PARSERS="$PARSERS $p"; done
[ -z "$PARSERS" ] && PARSERS="hermes"
PARSER_MAIN=$(awk '{print $1}' <<<"${PARSERS# }")
PARSER_ALT=$(awk '{print $2}'  <<<"${PARSERS# }"); PARSER_ALT="${PARSER_ALT:-$PARSER_MAIN}"
status "tool-parsers: '$PARSERS' → main=$PARSER_MAIN alt=$PARSER_ALT"

# ---------- 3. Cascada de arranque (el primero que responde /health gana) ----------
try_launch() {
  local label="$1" timeout="$2"; shift 2
  status "intento: $label"
  echo "########## $(date -u) :: $label" >> /workspace/vllm.log
  "$@" >> /workspace/vllm.log 2>&1 &
  local pid=$! waited=0
  while [ "$waited" -lt "$timeout" ]; do
    if ! kill -0 "$pid" 2>/dev/null; then
      status "murió tras ${waited}s: $label — siguiente escalón"; tail -4 /workspace/vllm.log; return 1
    fi
    if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" -o /dev/null 2>/dev/null; then
      echo "$label" > /workspace/WINNER; status "SIRVIENDO (${waited}s) :: $label"
      wait "$pid"; status "vLLM terminó (código $?) — contenedor sigue vivo"; return 0
    fi
    sleep 5; waited=$((waited+5))
  done
  status "timeout ${timeout}s sin /health: $label — matando"; kill -9 "$pid" 2>/dev/null; sleep 8; return 1
}

base_args() {
  echo python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" --api-key "$VLLM_API_KEY" \
    --gpu-memory-utilization "$GPU_UTIL" --trust-remote-code \
    --reasoning-parser qwen3 --enable-auto-tool-choice \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    ${CHAT_TEMPLATE:+--chat-template "$CHAT_TEMPLATE"} --uvicorn-log-level info \
    --max-num-seqs "$MAX_SEQS"
}

# FP8 + MTP (speculative decoding nativo, ~1.6x más rápido)
try_launch "fp8+mtp1 parser=$PARSER_MAIN ctx=$CTX" 1100 \
  $(base_args) --quantization fp8 --max-model-len "$CTX" --tool-call-parser "$PARSER_MAIN" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' && exit 0
# FP8 sin MTP (garantizado)
try_launch "fp8 parser=$PARSER_MAIN ctx=$CTX" 1000 \
  $(base_args) --quantization fp8 --max-model-len "$CTX" --tool-call-parser "$PARSER_MAIN" && exit 0
try_launch "fp8 parser=$PARSER_ALT ctx=$CTX" 1000 \
  $(base_args) --quantization fp8 --max-model-len "$CTX" --tool-call-parser "$PARSER_ALT" && exit 0
# BF16 puro (garantizado, más lento)
try_launch "bf16 parser=$PARSER_MAIN ctx=$CTX" 1000 \
  $(base_args) --max-model-len "$CTX" --tool-call-parser "$PARSER_MAIN" && exit 0
try_launch "bf16 parser=$PARSER_ALT ctx=$CTX" 1000 \
  $(base_args) --max-model-len "$CTX" --tool-call-parser "$PARSER_ALT" && exit 0
# Rescate: contexto a la mitad, CUDA graphs apagados
try_launch "RESCATE bf16 parser=$PARSER_ALT ctx=131072 eager" 1000 \
  $(base_args) --max-model-len 131072 --tool-call-parser "$PARSER_ALT" --enforce-eager && exit 0

status "FATAL: ninguna configuración levantó — ver /workspace/vllm.log"; tail -60 /workspace/vllm.log
sleep infinity
