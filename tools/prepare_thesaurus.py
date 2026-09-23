#!/usr/bin/env python3
"""Build word -> synonym JSONL from a LOCAL thesaurus source.

Accepted input:
  --moby TEXTFILE   Project Gutenberg Moby Thesaurus II raw text
                    (e.g. https://www.gutenberg.org/ebooks/3202 ;
                    public domain by express grant of Grady Ward, 2001.
                    If you keep Project Gutenberg headers/footers in a
                    redistributed file, follow their trademark rule and
                    remove PG references or keep the PG licence notice.)

Moby root lines start with a headword; following comma-separated lines
list synonyms. Pair-per-row output: context is the headword, options are
six unique words, label is one true synonym. Split by deterministic
headword hash. Standard library only.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def split_of(word):
    return int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 10


def read_moby(path):
    pairs = []
    head = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("***", "End of", "START OF")):
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z' .()-]{1,40}", line) and "," not in line:
            head = line.strip()
            continue
        if head and "," in line:
            for syn in line.split(","):
                syn = syn.strip()
                if syn and syn.casefold() != head.casefold():
                    pairs.append((head, syn))
    # de-duplicate directed pairs, drop self loops
    seen, unique = set(), []
    for word, syn in pairs:
        key = (word.casefold().strip(), syn.casefold().strip())
        if key[0] != key[1] and key not in seen:
            seen.add(key)
            unique.append((word, syn))
    return unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moby", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    pairs = read_moby(args.moby)
    pools = {"train": [], "validation": [], "test": []}
    for word, syn in pairs:
        m = split_of(word.casefold().strip())
        pools["train" if m < 8 else ("validation" if m == 8 else "test")].append(
            (word, syn))
    words = sorted({w for w, _ in pairs} | {s for _, s in pairs})
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, items in pools.items():
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for word, syn in items:
                negs = set()
                while len(negs) < args.negatives:
                    cand = rng.choice(words)
                    if cand.casefold().strip() not in (
                            word.casefold().strip(), syn.casefold().strip()):
                        negs.add(cand)
                options = [syn] + sorted(negs)
                rng.shuffle(options)
                fh.write(json.dumps({"context": word, "options": options,
                                     "label": options.index(syn)},
                                    ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {len(items)} rows sha256={digest}")


if __name__ == "__main__":
    main()
