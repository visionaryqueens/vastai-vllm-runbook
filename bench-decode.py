#!/usr/bin/env python3
"""bench-decode.py - measure pure decode throughput, correctly.

Two mistakes make almost every homemade LLM benchmark wrong, and this script
exists because both of them bit us:

1. **Measuring end-to-end.** total_tokens / wall_time folds time-to-first-token
   into the rate. TTFT here is ~600 ms; on a 200-token generation that is 25% of
   the wall clock and you under-report decode by a third. Time from the FIRST
   token to the LAST instead.
2. **Counting stream chunks as tokens.** With speculative decoding one chunk
   carries more than one token (1.87 on this model). Counting chunks under-reports
   by that same factor. Read `usage.completion_tokens` via `stream_options`.

Usage:  ./bench-decode.py ENDPOINT API_KEY MODEL [MAX_TOKENS]
"""
import json, sys, time, urllib.request
ep    = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
key   = sys.argv[2] if len(sys.argv) > 2 else ""
model = sys.argv[3] if len(sys.argv) > 3 else "qwen38-uncensored"
n     = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
label = f"max_tokens={n}"
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content":
        "Write a long detailed technical essay about GPU memory bandwidth and LLM inference."}],
    "max_tokens": n, "temperature": 0, "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
req = urllib.request.Request(
    f"{ep}/v1/chat/completions", data=body,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
t0 = time.time(); first = last = None; chunks = 0; real = None
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except Exception:
            continue
        if d.get("usage"):
            real = d["usage"].get("completion_tokens")
        ch = d.get("choices") or []
        delta = ch[0].get("delta") if ch else None
        if not delta or (not delta.get("content") and not delta.get("reasoning_content")):
            continue
        now = time.time()
        if first is None:
            first = now
        last = now; chunks += 1
ttft = (first - t0) if first else float("nan")
dec = (last - first) if (first and last and last > first) else float("nan")
tok = real if real else chunks
rate = (tok - 1) / dec if dec and dec > 0 else float("nan")
per = tok / chunks if chunks else float("nan")
print(f"  {label:<12} tokens={tok:<5} chunks={chunks:<5} ({per:.2f} tok/chunk)  "
      f"TTFT={ttft*1000:5.0f}ms  decode={rate:6.1f} tok/s")
