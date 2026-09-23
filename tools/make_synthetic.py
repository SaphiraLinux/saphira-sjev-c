#!/usr/bin/env python3
"""Independently authored synthetic SJEV training data (no external source).

Four task shapes: routing, colour/object selection, classification,
query -> passage choice. Deterministic given --seed. Emits train /
validation / test JSONL plus row counts and SHA256 hashes on stdout.
Standard library only.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

COLOURS = ("amber", "azure", "bronze", "coral", "crimson", "gold",
           "green", "indigo", "jade", "ochre")
OBJECTS = ("badger", "crane", "dolphin", "falcon", "gecko", "heron",
           "ibis", "jackal", "koala", "lynx")
DEPTS = ("returns", "sales", "support", "parts")
ITEMS = ("widget", "sprocket", "gizmo", "cog", "spring", "lever")
SYMPTOMS = ("rattles", "sticks", "overheats", "wobbles", "squeaks", "jams")
ITEM_DEPT = {"widget": "support", "sprocket": "parts", "gizmo": "support",
             "cog": "parts", "spring": "returns", "lever": "sales"}
PASSAGES = {
    "rain": "Rain falls when vapour cools and condenses into droplets.",
    "tides": "Tides rise and fall with the pull of the moon.",
    "compass": "A compass needle points north along magnetic lines.",
    "echo": "An echo returns when sound reflects off a hard wall.",
    "frost": "Frost forms when dew freezes on cold surfaces.",
    "thunder": "Thunder follows lightning as heated air expands.",
}
QUESTIONS = {
    "rain": "What makes rain fall?",
    "tides": "What pulls the tides?",
    "compass": "Where does a compass point?",
    "echo": "What causes an echo?",
    "frost": "When does frost form?",
    "thunder": "What causes thunder?",
}
SENTIMENTS = {
    "positive": ("I love this device.", "Great value, works well.",
                 "Fast delivery and friendly help."),
    "negative": ("It broke on day one.", "Terrible quality, avoid it.",
                 "Slow reply and rude service."),
    "neutral": ("The parcel arrived Tuesday.", "It weighs about two kilos.",
                "The manual has twelve pages."),
}


def routing(rng):
    item = rng.choice(ITEMS)
    dept = ITEM_DEPT[item]
    others = [d for d in DEPTS if d != dept]
    options = [dept] + rng.sample(others, 2)
    rng.shuffle(options)
    return {"context": f"Route the faulty {item}: it {rng.choice(SYMPTOMS)}. Desk?",
            "options": options, "label": options.index(dept)}


def colour_object(rng):
    target = f"{rng.choice(COLOURS)} {rng.choice(OBJECTS)}"
    pool = {target}
    while len(pool) < 4:
        pool.add(f"{rng.choice(COLOURS)} {rng.choice(OBJECTS)}")
    options = sorted(pool)
    rng.shuffle(options)
    return {"context": f"Pick the badge: {target}.",
            "options": options, "label": options.index(target)}


def classification(rng):
    label = rng.choice(("positive", "negative", "neutral"))
    return {"context": f"Classify: {rng.choice(SENTIMENTS[label])}",
            "options": ["positive", "negative", "neutral"],
            "label": ("positive", "negative", "neutral").index(label)}


def query_passage(rng):
    key = rng.choice(sorted(PASSAGES))
    others = [k for k in PASSAGES if k != key]
    picks = [key] + rng.sample(others, 3)
    rng.shuffle(picks)
    return {"context": QUESTIONS[key],
            "options": [PASSAGES[k] for k in picks],
            "label": picks.index(key)}


SHAPES = (routing, colour_object, classification, query_passage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--validation", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    total = args.train + args.validation + args.test
    for _ in range(total):
        rows.append(rng.choice(SHAPES)(rng))
    rng.shuffle(rows)
    splits = {"train": rows[:args.train],
              "validation": rows[args.train:args.train + args.validation],
              "test": rows[args.train + args.validation:]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {len(items)} rows sha256={digest}")


if __name__ == "__main__":
    main()
