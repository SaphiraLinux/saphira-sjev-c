#!/usr/bin/env python3
"""Export a TinyScorer .pt checkpoint to the sjev v1 binary format.

Usage:
  python src/export_model.py runs/wikispeedia-256-32.pt model.sjev
  python src/export_model.py runs/synthetic.pt --output model.sjev
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

MAGIC = b"SJEV"
VERSION = 1
HEADER_FMT = "<4sIIIIIII"  # magic, version, width, rank, context_tokens, option_tokens, r0, r1
HEADER_SIZE = 32


def _expected_state_keys():
    return [
        "embedding.weight",
        "position.weight",
        "head.context_norm.weight",
        "head.context_norm.bias",
        "head.option_norm.weight",
        "head.option_norm.bias",
        "head.query.weight",
        "head.key.weight",
        "head.value.weight",
    ]


def export_checkpoint(in_path: Path, out_path: Path) -> None:
    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    state = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(f"{in_path}: expected dict with 'config' and 'state_dict'")

    if config.get("encoder") != "tiny":
        raise ValueError(f"{in_path}: only encoder='tiny' can be exported (got {config.get('encoder')})")

    try:
        W = int(config["width"])
        R = int(config["rank"])
        C = int(config["context_tokens"])
        O = int(config["option_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{in_path}: config missing width/rank/context_tokens/option_tokens") from exc

    if not (1 <= W <= 4096 and 1 <= R <= 4096 and 1 <= C <= 8192 and 1 <= O <= 8192):
        raise ValueError(f"invalid dims: width={W} rank={R} context_tokens={C} option_tokens={O}")

    required = _expected_state_keys()
    missing = [k for k in required if k not in state]
    if missing:
        raise ValueError(f"{in_path}: missing keys: {missing}")

    # Validate shapes
    def _shape(t):
        return tuple(t.shape)

    checks = {
        "embedding.weight": (257, W),
        "position.weight": (C, W),
        "head.context_norm.weight": (W,),
        "head.context_norm.bias": (W,),
        "head.option_norm.weight": (W,),
        "head.option_norm.bias": (W,),
        "head.query.weight": (R, W),
        "head.key.weight": (R, W),
        "head.value.weight": (R, W),
    }
    for k, exp in checks.items():
        got = tuple(state[k].shape)
        if got != exp:
            raise ValueError(f"{in_path}: {k} shape {got} != expected {exp}")

    # Ensure all tensors are float32
    for k in required:
        v = state[k]
        if v.dtype != torch.float32:
            state[k] = v.float()

    header = struct.pack(HEADER_FMT, MAGIC, VERSION, W, R, C, O, 0, 0)
    assert len(header) == HEADER_SIZE

    total_floats = 257 * W + C * W + W + W + W + W + R * W + R * W + R * W
    expected_size = HEADER_SIZE + total_floats * 4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header)

        def _write_tensor(key: str):
            t = state[key].contiguous().cpu().float().numpy()
            # numpy tobytes is native LE on x86; file is defined LE, which matches host.
            # Ensure LE explicitly by checking dtype.
            f.write(t.astype("<f4").tobytes())

        for k in required:
            _write_tensor(k)

    actual = out_path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"wrote {actual} bytes but expected {expected_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TinyScorer checkpoint to sjev v1 binary")
    parser.add_argument("input", type=Path, help="input .pt checkpoint")
    parser.add_argument("output", type=Path, nargs="?", help="output .sjev path")
    parser.add_argument("--output", dest="output_opt", type=Path, help="alternative output path")
    args = parser.parse_args()

    out = args.output_opt if args.output_opt is not None else args.output
    if out is None:
        parser.error("output path required (positional or --output)")

    export_checkpoint(args.input, out)
    print(f"exported {args.input} -> {out}")


if __name__ == "__main__":
    main()
