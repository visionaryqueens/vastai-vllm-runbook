#!/usr/bin/env bash
# create-template.sh — crea el template público de Qwen3.8-27B en Vast.ai vía API.
# Corre en LOCAL. Lee la key de ~/.config/vastai/vast_api_key (nunca hardcodeada).
set -uo pipefail
unset VAST_API_KEY

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY="$(cat "$HOME/.config/vastai/vast_api_key")"
ONSTART="$(cat "$HERE/onstart-qwen38-vllm.sh")"

NAME="Qwen3.8-27B Uncensored — vLLM FP8 + MTP, 256K ctx"
DESC="Qwen3.8-27B abliterated (huihui) servido con vLLM: FP8 dinámico + MTP (~112 tok/s en H200), 256K ctx, tool-calling, vision. API OpenAI-compatible en :8000. Probado en H200 NVL 143GB; funciona en 80GB+ (H100, A100-80G)."
README='## Qwen3.8-27B Uncensored — vLLM FP8 + MTP

Modelo: `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (BF16, no gated) servido con vLLM.

### Qué obtenés
- API OpenAI-compatible en el puerto **8000** (Bearer key)
- **256K context** (262144 tokens)
- **Tool-calling** (auto tool choice) + **reasoning** (reasoning-parser qwen3)
- **Vision** (hasta 4 imágenes por prompt)
- **FP8 dinámico + MTP** (speculative decoding nativo): ~112 tok/s medido en H200 NVL
- Arranque en cascada: si FP8+MTP no levanta, prueba FP8, luego BF16, luego rescate a 128K — el primero que responde `/health` se queda

### Requisitos de hardware
- **Recomendado:** H200 NVL 143GB (probado) o H100 80GB
- **Mínimo:** 80GB VRAM (A100-80G). En 40GB no entra el modelo BF16.
- CUDA 12.0+

### Uso
1. Creá la instancia desde este template (el onstart descarga el modelo ~56GB y arranca vLLM solo; 5-20 min según red)
2. SSH a la instancia → `cat /workspace/STATUS` te da el estado y la **VLLM_API_KEY** (si no pasaste la tuya, se genera una)
3. Probá:
   ```
   curl http://localhost:8000/v1/chat/completions \
     -H "Authorization: Bearer $VLLM_API_KEY" \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"qwen38-uncensored\",\"messages\":[{\"role\":\"user\",\"content\":\"Hola\"}]}"
   ```
4. Desde tu máquina: túnel SSH `ssh -L 8000:localhost:8000 -p <puerto> root@<host>` y apuntá tu cliente a `http://localhost:8000/v1`

### Variables de entorno (opcionales)
- `VLLM_API_KEY` — tu Bearer key (si no la pasás, se genera y se muestra en /workspace/STATUS)
- `HF_TOKEN` — solo si cambiás a un repo gated vía `MODEL_REPO`
- `MODEL_REPO` — otro repo HF compatible (default: huihui-ai/Huihui-Qwen3.8-27B-abliterated)
- `CTX` — max-model-len (default 262144; bajalo a 131072 si la VRAM aprieta)
- `GPU_UTIL` — gpu-memory-utilization (default 0.92)
- `SERVED_NAME` — nombre servido (default qwen38-uncensored)

### Logs
- `/workspace/STATUS` — estado actual (una línea)
- `/workspace/onstart.log` — log completo del onstart
- `/workspace/vllm.log` — log de vLLM
- `/workspace/WINNER` — qué configuración de la cascada quedó sirviendo

### Nota de transparencia
Este template incluye el link de referido de su autor (Vast.ai referral program: 3% del gasto de por vida). Si lo creás desde el link de referido, el autor gana ese 3% — vos no pagás nada extra.
'

PAYLOAD=$(python3 - "$ONSTART" "$NAME" "$DESC" "$README" <<'PY'
import json, sys
onstart, name, desc, readme = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
print(json.dumps({
    "name": name,
    "image": "vllm/vllm-openai",
    "tag": "latest",
    "env": "-p 8000:8000",
    "onstart": onstart,
    "runtype": "ssh",
    "ssh_direct": True,
    "use_ssh": True,
    "private": False,
    "recommended_disk_space": 100,
    "extra_filters": {"cuda_max_good": {"gte": 12.0}},
    "desc": desc,
    "readme": readme,
    "href": "https://hub.docker.com/r/vllm/vllm-openai",
    "repo": "vllm/vllm-openai",
}))
PY
)

echo "Creando template: $NAME"
OUT=$(curl -sS -X POST "https://console.vast.ai/api/v0/template/" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo "$OUT" | jq .
ID=$(echo "$OUT" | jq -r '.template.id // .id // .template_id // empty' 2>/dev/null)
if [ -n "$ID" ]; then
  echo "$ID" > "$HERE/.template_id"
  echo "OK — template $ID creado (guardado en .template_id)"
  echo "Referral link: copiarlo en la UI → Templates → My Templates → tres puntos → Copy Referral Link"
else
  echo "ERROR: sin id en la respuesta"
  exit 1
fi
