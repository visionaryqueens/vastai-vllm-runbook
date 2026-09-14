#!/usr/bin/env bash
# verify.sh — prove a vLLM endpoint is actually usable, not just "running".
# Usage: ./verify.sh [ENDPOINT] [API_KEY]
#   ENDPOINT defaults to http://localhost:8000, API_KEY to $VLLM_API_KEY.
# Exits non-zero on the first failed check.
set -uo pipefail

EP="${1:-${VLLM_ENDPOINT:-http://localhost:8000}}"
KEY="${2:-${VLLM_API_KEY:-}}"
MODEL="${MODEL:-}"
AUTH=(); [ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")
fail(){ printf 'FAIL  %s\n' "$*" >&2; exit 1; }
ok(){ printf 'ok    %-12s %s\n' "$1" "$2"; }

# 1. health
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$EP/health") \
  || fail "health: no response from $EP"
[ "$code" = 200 ] || fail "health: HTTP $code (expected 200)"
ok health "200"

# 2. models — also discovers the served name if you didn't pass one
models=$(curl -s --max-time 15 "${AUTH[@]}" "$EP/v1/models") || fail "models: request failed"
read -r MODEL_FOUND CTX < <(printf '%s' "$models" | python3 -c '
import sys,json
try: d=json.load(sys.stdin)
except Exception: sys.exit("models: response is not JSON (bad API key?)")
m=(d.get("data") or [None])[0]
if not m: sys.exit("models: empty list")
print(m["id"], m.get("max_model_len","?"))') || fail "models: $(printf '%s' "$models" | head -c 200)"
MODEL="${MODEL:-$MODEL_FOUND}"
ok models "$MODEL (ctx $CTX)"

# 3. completion — thinking OFF, or the reasoning parser eats the whole budget
body=$(curl -s --max-time 60 "${AUTH[@]}" -H 'Content-Type: application/json' \
  "$EP/v1/chat/completions" -d "{
    \"model\":\"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly one word: READY\"}],
    \"max_tokens\":16,\"temperature\":0,
    \"chat_template_kwargs\":{\"enable_thinking\":false}}")
content=$(printf '%s' "$body" | python3 -c '
import sys,json
d=json.load(sys.stdin)
c=d["choices"][0]["message"].get("content")
print((c or "").strip())' 2>/dev/null) || fail "completion: bad response: $(printf '%s' "$body" | head -c 200)"
[ -n "$content" ] || fail "completion: empty content (reasoning parser ate the budget? raise max_tokens)"
ok completion "\"$content\""

# 4. tool call — the check that actually catches broken agentic serving
body=$(curl -s --max-time 90 "${AUTH[@]}" -H 'Content-Type: application/json' \
  "$EP/v1/chat/completions" -d "{
    \"model\":\"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"What time is it in Tokyo? Use the tool.\"}],
    \"max_tokens\":300,\"temperature\":0,
    \"chat_template_kwargs\":{\"enable_thinking\":false},
    \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",
      \"description\":\"Current time in a timezone\",
      \"parameters\":{\"type\":\"object\",
        \"properties\":{\"timezone\":{\"type\":\"string\",\"description\":\"e.g. Asia/Tokyo\"}},
        \"required\":[\"timezone\"]}}}]}")
call=$(printf '%s' "$body" | python3 -c '
import sys,json
d=json.load(sys.stdin)
m=d["choices"][0]["message"]
tc=m.get("tool_calls")
if not tc: sys.exit("no tool_calls; model replied: %r" % ((m.get("content") or "")[:120]))
f=tc[0]["function"]; print("%s(%s)" % (f["name"], f["arguments"]))' 2>&1) \
  || fail "tool call: $call"
ok "tool call" "$call"

printf '\nAll checks passed — %s is serving %s.\n' "$EP" "$MODEL"
