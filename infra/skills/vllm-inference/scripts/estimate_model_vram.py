#!/usr/bin/env python3
# Purpose: estimate vLLM weight + KV-cache VRAM from params, dtype, max-model-len; flag if it exceeds available VRAM.
# Usage: ./estimate_model_vram.py --params-b 7 --num-layers 32 --num-kv-heads 32 --head-dim 128 --max-model-len 4096 [--dtype bf16] [--vram-gb 48]
import argparse, shutil, subprocess, sys

# Bytes per element by dtype name (covers vLLM's common weight + KV-cache dtypes).
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
BYTES_PER_GIB = 1024 ** 3
KV_FACTOR = 2  # K and V are each cached, per layer, per token.

def probe_total_vram_gb():
    """Smallest per-GPU total memory (GiB) via nvidia-smi, or None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    mibs = [int(x) for x in out.stdout.split()]
    return min(mibs) / 1024  # MiB -> GiB; smallest GPU is the binding constraint.

def main():
    p = argparse.ArgumentParser(description="Estimate vLLM weight + KV-cache VRAM.")
    p.add_argument("--params-b", type=float, required=True, help="parameter count in billions")
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--num-kv-heads", type=int, required=True, help="KV heads (GQA-aware, not query heads)")
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--max-model-len", type=int, required=True, help="max context length in tokens")
    p.add_argument("--max-num-seqs", type=int, default=1, help="concurrent sequences sharing the KV cache")
    p.add_argument("--dtype", choices=DTYPE_BYTES, default="bf16", help="weight dtype")
    p.add_argument("--kv-dtype", choices=DTYPE_BYTES, help="KV-cache dtype (default: same as --dtype)")
    p.add_argument("--vram-gb", type=float, help="available VRAM per GPU (default: probe nvidia-smi)")
    a = p.parse_args()

    kv_dtype = a.kv_dtype or a.dtype
    weight_gb = a.params_b * 1e9 * DTYPE_BYTES[a.dtype] / BYTES_PER_GIB
    kv_per_tok = KV_FACTOR * a.num_layers * a.num_kv_heads * a.head_dim * DTYPE_BYTES[kv_dtype]
    kv_gb = kv_per_tok * a.max_model_len * a.max_num_seqs / BYTES_PER_GIB
    total_gb = weight_gb + kv_gb

    print(f"weights ({a.dtype}):   {weight_gb:7.2f} GiB")
    print(f"kv-cache ({kv_dtype}): {kv_gb:7.2f} GiB  ({a.max_model_len} tok x {a.max_num_seqs} seq)")
    print(f"total:               {total_gb:7.2f} GiB")

    vram = a.vram_gb if a.vram_gb is not None else probe_total_vram_gb()
    if vram is None:
        print("note: no --vram-gb and nvidia-smi unavailable; skipping fit check", file=sys.stderr)
        return 0
    print(f"available:           {vram:7.2f} GiB/GPU")
    if total_gb > vram:
        print(f"error: estimate exceeds VRAM by {total_gb - vram:.2f} GiB - quantize or shard", file=sys.stderr)
        return 1
    print("ok: fits, but leave headroom for activations + allocator fragmentation")
    return 0

if __name__ == "__main__":
    sys.exit(main())
