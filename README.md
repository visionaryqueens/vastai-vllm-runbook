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

## Which card should you actually rent?

`gpu-price-perf.py` pulls live Vast.ai offers, crosses them with vendor bandwidth
figures, and ranks by **decode tok/s per dollar** for your model size:

```
$ ./gpu-price-perf.py 14 --ctx 32000       # a 7B at bf16, 32k context

Model reads 14 GB/token, KV for 32,000 ctx ~6.4 GB -> needs ~24.4 GB VRAM

GPU                 $/hr    VRAM   GB/s   tok/s  tok/s per $
------------------------------------------------------------
Tesla V100         0.123     32G    900    39.2        319.4
RTX 5090           0.401     32G   1792    78.1        194.7
A100 SXM4          0.428     40G   1555    67.8        158.3
Q RTX 8000         0.254     48G    672    29.3        115.2
A100 PCIE          1.001     80G   1935    84.3         84.2
RTX 6000Ada        0.601     48G    960    41.8         69.6 *
```

The result is usually not the card you expected. A **Tesla V100 at $0.12/hr delivers
~4x the tok/s per dollar of an RTX 5090**, because decode only cares about bandwidth
and the V100 still has 900 GB/s of it. If you want raw speed the 5090 wins; if you
want throughput per dollar, the boring old card does.

That is exactly why the tool prints dtype caveats alongside the ranking — a V100 is
Volta, so no bf16, no fp8, no FlashAttention-2. **A card can top the table and still
refuse to load your quant.** Read that column first, price second.

For MoE models pass the *active* parameter bytes, not the checkpoint size — decode
only reads the experts it routes to.

## The formula, validated

`tok/s ≈ bandwidth × efficiency / GB_per_token`.

**The efficiency constant depends on the engine, and that matters more than it sounds.**
Fitting it against llama.cpp gives ~0.61. We then measured the same formula against vLLM
and it was off by 16% — vLLM simply extracts more of the available bandwidth, at ~0.71.
Publishing one constant for both would quietly mispredict every vLLM number on this page.

| Card | Engine | GB of weights | Predicted | Measured | Error |
|---|---|---|---|---|---|
| RTX 3090 | llama.cpp | 17.4 (Q4) | 32.8 | **33.3** | +1.5% |
| RTX 6000 Ada | llama.cpp | 29.0 (Q8) | 20.2 | **20.1** | −0.5% |
| H200 NVL | llama.cpp | 35.0 (fp8) | 83.7 | **83.5** | +0.2% |
| H100 NVL | **vLLM** | 29.8 (fp8) | 80.4 | **80.4** | <0.1% |

Five architectures, two engines, under 2% error once the constant matches the engine.
It is a back-of-envelope model, not a benchmark — but it is good enough to pick hardware
before you spend anything.

Note those are all **without** speculative decoding. With MTP enabled the same H100 NVL
does 129.6 tok/s, because each decode step emits two tokens instead of one — spec decoding
beats the bandwidth bound rather than obeying it. Size hardware with the formula, then
treat MTP as upside.

## What actually breaks

Things I hit on real instances that no doc warned me about:

**MTP dies at startup because of a default that is 41 too high.** For three boots in a
row the `fp8 + mtp` launch died and the cascade fell through to plain `fp8`. It reads like
an unsupported-feature problem. It isn't:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (983).
Each decode sequence requires one Mamba cache block, so CUDA graph capture
cannot proceed. Please lower max_num_seqs to at most 983.
```

`1024` is vLLM's default. A hybrid model needs one Mamba cache block per decode sequence
and there are only 983 to go around, so speculative decoding can never capture its graphs.
Pass **`--max-num-seqs 512`** and MTP starts. Measured on an H100 NVL, same card, same
price, same 256K context:

| | decode |
|---|---|
| `fp8`, default `max_num_seqs` | 80.4 tok/s |
| `fp8 + mtp`, `--max-num-seqs 512` | **129.6 tok/s** |

**+61%** from one flag. vLLM reports `Mean acceptance length: 2.00` with a 100% draft
acceptance rate on this model — every decode step emits two tokens. Serving a single user
never needed 1024 concurrent sequences anyway.

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

**Prefix caching on hybrids: the advice you'll find is out of date.** Qwen3.5/3.8
(`qwen3_5`) is hybrid — only 16 of 64 layers carry KV — and for a long time that meant no
prompt reuse: Ollama logs `forcing full prompt re-processing`, and `--enable-prefix-caching`
was a silent no-op on Mamba-2/GDN hybrids. At 256K that cost ~150 s *per message*, which is
why every guide (including an earlier version of this one) tells you to cap context at
64–128K for agentic work.

**vLLM 0.29 fixed it.** The engine now selects a Mamba cache mode that is compatible with
prefix caching, and the hit rate is real:

```
Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration
when prefix caching is enabled
...
Prefix cache hit rate: 95.7% / 95.8% / 96.1%
```

So long context stopped being something you re-pay every turn, and the 64–128K ceiling no
longer applies. Check your own logs before believing either version of this advice — grep
for `Prefix cache hit rate` and let the number decide.

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
| `gpu-price-perf.py` | Ranks live Vast offers by decode tok/s per dollar for your model size, with dtype caveats. No API key read or stored. |
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
