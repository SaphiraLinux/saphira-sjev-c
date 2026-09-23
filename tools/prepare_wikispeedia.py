#!/usr/bin/env python3
"""Build Wikispeedia-style next-click JSONL from a user-supplied SNAP archive.

Upstream: Stanford SNAP Wikispeedia
  https://snap.stanford.edu/data/wikispeedia.html
  Robert West and Jure Leskovec, Human Wayfinding in Information Networks,
  WWW 2012. Cite West/Pineau/Precup IJCAI 2009 for the game data.
No dataset licence is granted on that page (the SNAP BSD licence covers
library code only); article text remains CC BY-SA/GFDL per Wikipedia.
This script therefore consumes a LOCALLY downloaded archive and never
redistributes one. Expected input layout (the two plaintext tarballs):

  <root>/wikispeedia_paths-and-graph/links.tsv
  <root>/wikispeedia_paths-and-graph/paths_finished.tsv
  <root>/plaintext_articles/<Article>.txt

Output: train/validation/test JSONL with a deterministic split by stable
hash of the target article (no leakage of a target across splits).
Standard library only.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
from urllib.parse import unquote


def _stable(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _title(text):
    return unquote(text).replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-options", type=int, default=16)
    args = ap.parse_args()
    graph = args.root / "wikispeedia_paths-and-graph"
    outgoing = {}
    for line in (graph / "links.tsv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            source, target = line.split("\t")
            outgoing.setdefault(source, []).append(target)
    buckets = {"train": [], "validation": [], "test": []}
    lines = (graph / "paths_finished.tsv").read_text(encoding="utf-8").splitlines()
    for row, line in enumerate(lines):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        path = []
        for node in fields[3].split(";"):
            if node == "<":
                if len(path) > 1:
                    path.pop()
            else:
                path.append(node)
        if len(path) < 2:
            continue
        step = _stable(f"{fields[0]}:{fields[1]}:{row}") % (len(path) - 1)
        current, click, target = path[step], path[step + 1], path[-1]
        candidates = list(dict.fromkeys(outgoing.get(current, ())))
        if len(candidates) < 2 or click not in candidates:
            continue
        rng = random.Random(_stable(f"{row}:{target}:menu"))
        others = [c for c in candidates if c != click]
        rng.shuffle(others)
        menu = [click] + others[:args.max_options - 1]
        rng.shuffle(menu)
        article = args.root / "plaintext_articles" / f"{current}.txt"
        try:
            body = " ".join(article.read_text(encoding="utf-8",
                                              errors="replace").split())
        except FileNotFoundError:
            continue
        record = {
            "context": f"Target article: {_title(target)}\n"
                       f"Current article: {_title(current)}\n{body[:2048]}",
            "options": [_title(m) for m in menu],
            "label": menu.index(click),
        }
        bucket = _stable(target + ":split") % 10
        split = "test" if bucket == 0 else ("validation" if bucket == 1 else "train")
        buckets[split].append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in buckets.items():
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {len(records)} rows sha256={digest}")


if __name__ == "__main__":
    main()
