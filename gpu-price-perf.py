#!/usr/bin/env python3
"""
gpu-price-perf.py - rank live Vast.ai GPU offers by decode throughput per dollar.

Decode is memory-bound: each token reads the weights out of VRAM, so throughput
tracks memory bandwidth, not VRAM size. VRAM only decides whether the model fits.

    tok/s ~= bandwidth_GB/s * EFFICIENCY / GB_read_per_token

EFFICIENCY (0.61) is measured, not assumed - see MEASUREMENTS below and the
validation table in the README.

Usage:
    ./gpu-price-perf.py 30              # 30 GB of weights (e.g. a 27B at fp8)
    ./gpu-price-perf.py 30 --ctx 128000 # reserve KV cache headroom too
    ./gpu-price-perf.py 14 --top 15     # a 7B at bf16, show 15 rows

Requires the `vastai` CLI, authenticated. No API key is read or stored by this script.
"""
import argparse, json, shutil, subprocess, sys

EFFICIENCY = 0.61

# Memory bandwidth in GB/s, from vendor specs. Only cards whose figure is
# unambiguous are listed; anything absent is skipped rather than guessed.
BANDWIDTH = {
    "GTX 1080": 320, "GTX 1080 Ti": 484, "RTX 2070": 448, "RTX 2070S": 448,
    "RTX 2080 Ti": 616, "Titan RTX": 672, "Q RTX 8000": 672,
    "Tesla P100": 732, "Tesla V100": 900,
    "RTX 3060": 360, "RTX 3060 Ti": 448, "RTX 3070": 448, "RTX 3070 Ti": 608,
    "RTX 3080": 760, "RTX 3080 Ti": 912, "RTX 3090": 936,
    "RTX A2000": 288, "RTX A4000": 448, "A10": 600, "A40": 696, "L4": 300,
    "RTX 4060": 272, "RTX 4060 Ti": 288, "RTX 4070": 504, "RTX 4070S": 504,
    "RTX 4070 Ti": 504, "RTX 4070S Ti": 504, "RTX 4080": 717, "RTX 4080S": 736,
    "RTX 4090": 1008, "RTX 4000Ada": 360, "RTX 4500Ada": 432, "RTX 6000Ada": 960,
    "L40S": 864,
    "A100 PCIE": 1935, "A100 SXM4": 1555,
    "RTX 5070": 672, "RTX 5070 Ti": 896, "RTX 5080": 960, "RTX 5090": 1792,
    "H100 PCIE": 2039, "H100 SXM": 3350, "H100 NVL": 3369,
    "H200 NVL": 4800, "H200 SXM": 4800,
}

# Dtype/architecture caveats. Cheap old cards often win on tok/s per dollar but
# cannot run the format you were planning to serve - flag that before someone rents one.
CAVEATS = {
    "Tesla P100": "Pascal: no bf16, no fp8, no FlashAttention. fp16 only.",
    "Tesla V100": "Volta: no bf16, no fp8, no FA2. fp16 only - check your stack supports it.",
    "GTX 1080": "Pascal: fp16 is emulated and slow. Practically fp32 only.",
    "GTX 1080 Ti": "Pascal: fp16 is emulated and slow. Practically fp32 only.",
    "Q RTX 8000": "Turing: no bf16, no fp8. fp16 only.",
    "Titan RTX": "Turing: no bf16, no fp8. fp16 only.",
    "RTX 2070": "Turing: no bf16, no fp8. fp16 only.",
    "RTX 2070S": "Turing: no bf16, no fp8. fp16 only.",
    "RTX 2080 Ti": "Turing: no bf16, no fp8. fp16 only.",
    "A100 PCIE": "Ampere: bf16 yes, fp8 no (needs Ada/Hopper+).",
    "A100 SXM4": "Ampere: bf16 yes, fp8 no (needs Ada/Hopper+).",
    "A40": "Ampere: bf16 yes, fp8 no.",
    "A10": "Ampere: bf16 yes, fp8 no.",
    "RTX 3090": "Ampere: bf16 yes, fp8 no.",
}

# Our own end-to-end measurements on rented cards: (GB of weights, tok/s observed).
# These are what EFFICIENCY was fitted to; the README shows predicted vs actual.
MEASUREMENTS = {
    "RTX 3090":    (17.4, 33.3),
    "RTX 6000Ada": (29.0, 20.1),
    "H100 NVL":    (29.8, 70.0),
    "H200 NVL":    (35.0, 83.5),
}


def fetch_offers():
    if not shutil.which("vastai"):
        sys.exit("vastai CLI not found. pip install vastai, then `vastai set api-key ...`")
    try:
        raw = subprocess.run(
            ["vastai", "search", "offers", "rentable=true num_gpus=1", "-o", "dph", "--raw"],
            capture_output=True, text=True, timeout=90, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        sys.exit(f"vastai search failed: {(e.stderr or '').strip()[:300]}")
    except subprocess.TimeoutExpired:
        sys.exit("vastai search timed out")
    raw = "".join(c for c in raw if c >= " " or c == "\n")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit("could not parse vastai output as JSON")
    return data if isinstance(data, list) else data.get("offers", [])


def main():
    ap = argparse.ArgumentParser(description="Rank Vast.ai offers by decode tok/s per dollar.")
    ap.add_argument("weights_gb", type=float, help="GB read per token (weights; for MoE use ACTIVE params)")
    ap.add_argument("--ctx", type=int, default=0, help="context length to reserve KV headroom for")
    ap.add_argument("--kv-gb-per-100k", type=float, default=20.0,
                    help="GB of KV per 100k tokens, model-specific (default 20, measured on a 27B)")
    ap.add_argument("--top", type=int, default=12, help="rows to show")
    args = ap.parse_args()

    kv_gb = (args.ctx / 100_000) * args.kv_gb_per_100k if args.ctx else 0.0
    needed = args.weights_gb + kv_gb + 4.0  # +4 GB activations/graphs headroom

    cheapest = {}
    for o in fetch_offers():
        name, dph, vram = o.get("gpu_name"), o.get("dph_total"), o.get("gpu_ram")
        if not name or not dph or not vram:
            continue
        if name not in cheapest or dph < cheapest[name]["dph"]:
            cheapest[name] = {"dph": dph, "vram_gb": vram / 1024}

    rows = []
    for name, o in cheapest.items():
        bw = BANDWIDTH.get(name)
        if not bw or o["vram_gb"] < needed:
            continue
        toks = bw * EFFICIENCY / args.weights_gb
        rows.append({
            "name": name, "dph": o["dph"], "vram": o["vram_gb"], "bw": bw,
            "toks": toks, "per_dollar": toks / o["dph"],
            "measured": name in MEASUREMENTS,
            "caveat": CAVEATS.get(name),
        })
    rows.sort(key=lambda r: -r["per_dollar"])

    if not rows:
        print(f"No available card fits {needed:.1f} GB right now "
              f"(weights {args.weights_gb} + KV {kv_gb:.1f} + 4 overhead).")
        return

    print(f"\nModel reads {args.weights_gb:g} GB/token"
          + (f", KV for {args.ctx:,} ctx ~{kv_gb:.1f} GB" if args.ctx else "")
          + f" -> needs ~{needed:.1f} GB VRAM\n")
    print(f"{'GPU':<16}{'$/hr':>8}{'VRAM':>8}{'GB/s':>7}{'tok/s':>8}{'tok/s per $':>13}")
    print("-" * 60)
    for r in rows[: args.top]:
        star = " *" if r["measured"] else ""
        print(f"{r['name']:<16}{r['dph']:>8.3f}{r['vram']:>7.0f}G{r['bw']:>7.0f}"
              f"{r['toks']:>8.1f}{r['per_dollar']:>13.1f}{star}")
    if any(r["measured"] for r in rows[: args.top]):
        print("\n* tok/s validated against our own measurement on this card.")
        print("Others are the formula's estimate: bandwidth x 0.61 / GB-per-token.")
    else:
        print("\nAll figures are the formula's estimate: bandwidth x 0.61 / GB-per-token.")
    print("MoE models read only ACTIVE params per token - pass that, not total size.")

    shown = rows[: args.top]
    noted = [r for r in shown if r["caveat"]]
    if noted:
        print("\nBefore you rent - dtype support is not the same across these:")
        for r in noted:
            print(f"  {r['name']:<14} {r['caveat']}")
        print("  A card can top the table on tok/s per dollar and still be unable to")
        print("  load your quant. Check this column first, price second.")
    print()


if __name__ == "__main__":
    main()
