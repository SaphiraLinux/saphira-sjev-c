#!/usr/bin/env python3
"""Build definition -> word JSONL from a LOCAL dictionary source.

Accepted inputs (one required):
  --opted TEXTFILE   Project Gutenberg OPTED / Webster 1913 plain text
                     (public domain; e.g. the OPTED archive mirrored at
                     https://sourceforge.net/projects/mysqlenglishdictionary)
  --sql DUMPFILE     A MySQL dump with (word, wordtype, definition) rows,
                     e.g. https://github.com/akadata/word-dictionary
                     (NOTE: that repository carries NO licence metadata, so
                     redistribution of its dump is NOT assumed; use it only
                     as a local input and check provenance yourself)

Output: train/validation/test JSONL, split by deterministic word hash so
all senses of one headword stay in a single split. Context is the
definition, options are six unique words, label is the defined word.
Standard library only.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def split_of(word):
    return int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 10


def read_sql(path):
    rows = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(
            r"\(\s*'((?:[^'\\]|\\.)*)'\s*,\s*'((?:[^'\\]|\\.)*)'\s*,"
            r"\s*'((?:[^'\\]|\\.)*)'\s*\)", text):
        word = m.group(1).replace("\\'", "'").strip()
        definition = m.group(3).replace("\\'", "'").replace("\\n", " ").strip()
        if word and len(definition) >= 30:
            rows.append((word, definition))
    return rows


def read_opted(path):
    rows = []
    word, buf = None, []
    for line in Path(path, encoding="utf-8", errors="replace").read_text().splitlines():
        head = re.match(r"^([A-Z][A-Za-z' -]{1,24})$", line.rstrip())
        if head and buf and word:
            definition = " ".join(buf).strip()
            if len(definition) >= 30:
                rows.append((word, definition))
            buf = []
        if head:
            word = head.group(1).strip()
        elif word and line.strip():
            buf.append(line.strip())
    return rows


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--opted", type=Path)
    src.add_argument("--sql", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rows = read_opted(args.opted) if args.opted else read_sql(args.sql)
    pools = {"train": [], "validation": [], "test": []}
    for word, definition in rows:
        m = split_of(word.casefold().strip())
        pools["train" if m < 8 else ("validation" if m == 8 else "test")].append(
            (word, definition))
    words = sorted({w for w, _ in rows})
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, items in pools.items():
        path = args.output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for word, definition in items:
                negs = set()
                while len(negs) < args.negatives:
                    cand = rng.choice(words)
                    if cand.casefold().strip() != word.casefold().strip():
                        negs.add(cand)
                options = [word] + sorted(negs)
                rng.shuffle(options)
                fh.write(json.dumps({"context": definition, "options": options,
                                     "label": options.index(word)},
                                    ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {len(items)} rows sha256={digest}")


if __name__ == "__main__":
    main()
