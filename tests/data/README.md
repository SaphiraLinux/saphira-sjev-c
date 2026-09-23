# Public test data and dataset provenance

All files here are redistributable. Nothing is copied from private corpora.

## Committed fixtures (generated or hand-authored in this repository)

- `train.jsonl` (160 rows), `validation.jsonl` (32), `test.jsonl` (32):
  produced by `tools/make_synthetic.py --output-dir tests/data --train 160
  --validation 32 --test 32 --seed 7`. Independently authored content
  (routing, colour/object, classification, query/passage shapes).
- `dictionary_mini.jsonl` (24 rows): hand-authored plain-English
  word/definition pairs, written for this repository in 2026. No upstream
  dictionary text is included.
- `thesaurus_mini.jsonl` (24 rows): hand-authored common synonym pairs,
  written for this repository in 2026. No upstream thesaurus text included.
- `wikispeedia_mini.jsonl` (24 rows): hand-authored invented article titles
  and one-sentence entries in navigation-menu shape. No Wikipedia text.

Regenerate the synthetic splits any time with the command above; hashes
must then match the values recorded in `results/` runs that used them.

## External sources (never committed; use the prepare scripts)

- Wikispeedia (SNAP, West & Leskovec, WWW 2012):
  https://snap.stanford.edu/data/wikispeedia.html — no dataset licence is
  granted there (SNAP BSD covers code only); article text is CC BY-SA/GFDL.
  Use `tools/prepare_wikispeedia.py --root <download> --output <dir>`.
- Dictionary: `tools/prepare_dictionary.py --opted <gutenberg-text>` (OPTED /
  Webster 1913 plain text, public domain) or `--sql <local-dump>`. The fork
  https://github.com/akadata/word-dictionary carries NO licence metadata,
  so its dump must not be redistributed; use it only as local input.
- Thesaurus: `tools/prepare_thesaurus.py --moby <gutenberg-3202-text>`
  (Moby Thesaurus II, public domain by express grant of Grady Ward, 2001;
  keep or remove Project Gutenberg branding per their trademark rule).
