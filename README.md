# vastai-vllm-runbook

Rent a GPU by the second, serve a 27B model with vLLM, tear it down when you're done.

This is the actual runbook — the `onstart` script that boots the server unattended, a
verifier that proves tool-calling works, and the numbers I measured on rented hardware.
No affiliation with Vast.ai beyond being a customer (see [disclosure](#disclosure)).

## Quick start

One-click template (vLLM + Qwen3.8-27B, OpenAI-compatible API on `:8000`):

**→ [Launch the template](https://cloud.vast.ai/?ref_id=580840&template_id=c00f9084cb157d729db47e4aa9a9cf08)**

Or do it by hand:

```bash
vastai search offers 'gpu_name=H200_NVL rentable=true' -o dph
vastai create instance <ID> --image vllm/vllm-openai:latest --disk 100 \
  --onstart-cmd "$(cat onstart-qwen38-vllm.sh)"
# ... 5-20 min later:
./verify.sh http://localhost:8000 "$VLLM_API_KEY"
vastai stop instance <ID>     # billing stops. storage still bills — destroy if you're done.
```

## Pick the GPU by memory bandwidth, not VRAM

This is the part people get wrong and it costs them real money.

Decode is **memory-bound**: every token read the whole weight set out of VRAM. So:

```
tok/s  ≈  bandwidth_GB/s  ×  0.61  /  GB_of_weights
```

`0.61` is the efficiency I measured across two different rented GPUs with llama.cpp.
VRAM only decides **whether the model fits**. Bandwidth decides **how fast it runs**.

| GPU | BW GB/s | $/hr | Q4 (17.4 GB) | Q8 (29 GB) |
|---|---|---|---|---|
| RTX 3090 24G | 936 | 0.148 | **33.3 measured** | doesn't fit |
| RTX 6000 Ada 48G | 960 | 0.619 | ~34 | **20.1 measured** |
| RTX 5090 32G | 1458 | 0.35–0.48 | ~51 | doesn't fit |
| A100 40G | 1315 | 0.54–0.67 | ~46 | ~28 (KV q8) |
| H100 NVL 94G | 3369 | 2.55 | ~118 | ~71 |
| H200 NVL 143G | 4800 | 3.81 | ~142 | **83.5 measured** |
| B200 179G | 6173 | 5.88 | ~216 | ~130 |

**The trap:** an RTX 6000 Ada is *not* faster than a 3090 — 960 vs 936 GB/s, within noise —
and costs 4.2×. You are buying VRAM headroom, not speed. Only pay for it if the model
genuinely doesn't fit.

**The other trap:** going Q4 → Q8 costs ~40% of your tok/s, because you nearly doubled
the bytes read per token. Worth it for quality, but know what you're paying.

## What actually breaks

Things I hit on real instances that no doc warned me about:

**MTP (speculative decoding) is not guaranteed to start.** On an H100 NVL the
`fp8 + mtp` launch died after 70s; plain `fp8` came up in 90s. That's why
`onstart-qwen38-vllm.sh` boots as a **cascade** — fp8+mtp → fp8 → bf16 → 128K rescue —
and the first config to answer `/health` wins and holds the container. Losing MTP costs
you roughly 112 → 70 tok/s. It still serves.

**`--reasoning-parser qwen3` will eat your whole token budget.** Ask for 20 tokens and
you get `content: null` with `reasoning_tokens: 20` — the model spent every token thinking
and never emitted an answer. For agentic use, send:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

**Vast maps each container port to a *different* host port.** `22/tcp` and `8000/tcp`
do not get consecutive numbers, and `direct_port_start` is not necessarily SSH. I lost
time SSH-ing into the vLLM port and getting `kex_exchange_identification: Connection
closed`. Read the real mapping:

```bash
vastai show instances-v1 --raw | jq '.instances[0].ports'
```

**Hybrid-attention models don't reuse prompt cache.** Qwen3.5/3.8 (`qwen3_5`) is hybrid —
only 16 of 64 layers carry KV. Ollama logs `forcing full prompt re-processing`, and vLLM's
`--enable-prefix-caching` is a documented silent no-op on Mamba-2/GDN hybrids. Every turn
re-processes the entire context. At 256K that's ~150 s **per message**. So:
- **Agentic** (many turns): 64–128K is the useful ceiling.
- **One-shot** (big corpus, few questions): 256K pays off, you eat the prefill once.

**A stopped instance still bills storage.** `stop` pauses GPU billing and keeps your disk.
With a few 100 GB instances parked that's $2–3/day quietly accruing. `destroy` when done.

**Never bind the API to `0.0.0.0` without `--api-key`.** Normal access is an SSH tunnel:

```bash
ssh -N -L 18000:localhost:8000 -p <ssh_port> root@<host>
```

The published port is a fallback, and only with a Bearer key set.

## Files

| File | What it does |
|---|---|
| `onstart-qwen38-vllm.sh` | Unattended boot: downloads the model, patches the chat template to allow multiple system messages, detects which tool-parser the build accepts, then launches the cascade. Logs to `/workspace/{STATUS,onstart.log,vllm.log,WINNER}`. |
| `verify.sh` | Proves the endpoint is real: `/health`, `/v1/models`, a completion with thinking off, and an actual tool call. Exits non-zero if any step fails. |
| `create-template.sh` | Publishes the whole thing as a Vast.ai template via the API. Reads your key from `~/.config/vastai/vast_api_key` — never hardcoded. |

## Verifying, not assuming

`verify.sh` exists because "the container is running" is not the same as "the model
answers tool calls". The check that matters:

```bash
./verify.sh https://<host>:<port> "$VLLM_API_KEY"
# health ....... 200
# models ....... qwen38-uncensored (ctx 262144)
# completion ... "READY"
# tool call .... get_time({"timezone": "Asia/Tokyo"})   <- the one that catches real breakage
```

## Disclosure

The template link above carries my Vast.ai referral ID. Vast's program pays the referrer
3% of what referred accounts spend, as account credit. **It does not change your price** —
there's no markup, and the same template works if you strip the `ref_id`. I use these
GPUs myself; the numbers in this README are from my own invoices.

## License

MIT — see [LICENSE](LICENSE).
