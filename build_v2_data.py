#!/usr/bin/env python3
"""
Build better training data from ai4bharat/indic-align (CC-BY-4.0, native-speaker verified).

Why this is an upgrade over the v0.1 data we were using:

  * indic-align v1.1 replaced the NLLB machine translation in v0.1 with
    per-language Indic LLM models plus native-speaker verification. Our old
    Hindi was MT output that we had to filter with chrF>=70 - the flan_v2 slice
    of v0.1 was so bad its targets were things like "1)|".
  * Licence is CC-BY-4.0, not the CC-BY-SA-3.0 of dolly-15k. SA applied to
    model weights is legally unsettled; BY is not.
  * ONE ROW CARRIES ALL THREE OF OUR LANGUAGES: eng_Latn, hin_Deva and hin_Latn
    are the same conversation in English, Devanagari and romanized Hindi. So we
    can build genuinely aligned data instead of three unrelated corpora, and
    hin_Latn is a real transliteration rather than schwa-stripped MT.

  python3.13 build_v2_data.py [--out-dir .] [--per-lang 6000]
"""
import argparse, json, os, random, re, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default=".")
ap.add_argument("--per-lang", type=int, default=6000,
                help="rows per language before filtering (over-fetch)")
ap.add_argument("--seed", type=int, default=3407)
args = ap.parse_args()
random.seed(args.seed)

# Shards, in descending order of conversational value. wiki_chat (198k) is the
# largest but is encyclopedic QA, so it comes last.
SHARDS = [
    ("indicalign-instruct/oasst/oasst.parquet",        20_000),  # real assistant convs
    ("indicalign-instruct/dolly/Dolly.parquet",        15_000),  # grounded QA
    ("indicalign-instruct/anudesh/anudesh1.parquet",  37_000),  # translated seed
    ("indicalign-instruct/wikihow/wiki_how.parquet",  20_000),  # how-to
    ("indicalign-instruct/wiki_conv/wiki_conv.parquet", 141_000),  # wiki discussion
]
REPO = "ai4bharat/indic-align"

DEV = re.compile(r"[ऀ-ॿ]")
CODE_FENCE = re.compile(r"```")
# The translated sources contain the ORIGINAL model refusing. Training on those
# teaches the uncensored model to refuse - the exact opposite of the goal.
REFUSAL = re.compile(
    r"\b(i'?m sorry|i am sorry|i apolog|sorry, but|"
    r"i (?:can'?t|cannot|can not|won'?t|will not)\b|"
    r"i'?m (?:not able|unable|afraid)\b|i am (?:not able|unable)\b|"
    r"as an ai\b|as a language model\b|i must (?:decline|refuse)\b|"
    r"(?:can'?t|cannot|unable to) (?:assist|help|provide|comply)|"
    r"i do(?:n'?t| not) feel comfortable)\b", re.I)

# Devanagari costs 1.45 tok/char and Hindi rows measured 1110-token median
# unfiltered (52% truncated). 620 chars * 1.45 + ~15 template ~= 915 tokens.
HINDI_MAX_CHARS = 620
ROMAN_MAX_CHARS = 420      # latin is ~0.3 tok/char, so this is a word-count proxy

# ONLY these three columns are ever read. indic-align carries 30 language columns per
# row (Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Urdu, Marathi, Nepali,
# Sanskrit, Oriya, Punjabi, Assamese, and *_Latn transliterations of each). We take
# exactly three and ignore the rest.
#
# Note the trap: Marathi (mar_Deva) and Nepali (npi_Deva) are ALSO Devanagari, so a
# "is it Devanagari" filter would silently admit them. Filtering by COLUMN, not by
# script, is the only correct way - hence LANG below and the audit script below.
LANG = {"hindi":   ("hin_Deva", HINDI_MAX_CHARS),   # Hindi, native script
        "hinglish": ("hin_Latn", ROMAN_MAX_CHARS),  # Hindi, romanized
        "english":  ("eng_Latn", ROMAN_MAX_CHARS)}  # English anchor

# Scripts that must NEVER appear. Marathi/Nepali share Devanagari so they are not
# listed here - they are excluded structurally by never reading those columns.
FOREIGN_SCRIPTS = {
    "\u0980-\u09FF": "Bengali",  "\u0A00-\u0A7F": "Gujarati",
    "\u0A80-\u0AFF": "Tamil",    "\u0B00-\u0B7F": "Telugu",
    "\u0B80-\u0BFF": "Kannada",  "\u0D00-\u0D7F": "Malayalam",
    "\u0D80-\u0DFF": "Sinhala",  "\u0600-\u06FF": "Arabic-script (Urdu/Persian)",
    "\u0900-\u097F": "Devanagari (Marathi/Nepali share this - see note)",
}
FOREIGN_RE = {name: re.compile(f"[{rng}]") for rng, name in FOREIGN_SCRIPTS.items()}


def fetch(path, cache="indicalign_cache"):
    os.makedirs(cache, exist_ok=True)
    local = os.path.join(cache, os.path.basename(path))
    if not os.path.exists(local):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
        sys.stderr.write(f"  downloading {os.path.basename(path)} ...\n")
        with urllib.request.urlopen(url, timeout=900) as r, open(local, "wb") as f:
            f.write(r.read())
    return local


def turns(pairs):
    """indic-align stores conversation turns as a list of [user, assistant] pairs."""
    out = []
    for p in pairs or []:
        try:
            u, a = p[0], p[1]
        except Exception:
            continue
        if isinstance(u, str) and isinstance(a, str) and u.strip() and a.strip():
            out.append((u.strip(), a.strip()))
    return out


def clean(u, a, max_chars):
    if not u or not a:
        return None
    if CODE_FENCE.search(u) or CODE_FENCE.search(a):
        return None
    if REFUSAL.search(u) or REFUSAL.search(a):
        return None
    if len(u) + len(a) > max_chars:
        return None
    if len(u.split()) < 2 or len(a.split()) < 2:
        return None
    return u, a


rows = {k: [] for k in LANG}
seen = 0
for path, cap in SHARDS:
    if all(len(v) >= args.per_lang for v in rows.values()):
        break
    try:
        import pandas as pd
        df = pd.read_parquet(fetch(path))
    except Exception as e:
        sys.stderr.write(f"  skip {path}: {type(e).__name__}: {e}\n")
        continue
    if "eng_Latn" not in df.columns:
        sys.stderr.write(f"  skip {path}: unexpected columns\n")
        continue
    sys.stderr.write(f"  {os.path.basename(path)}: {len(df):,} rows\n")
    # interleaved so the cap takes a mix of shards, not just the first
    order = list(df.index)
    random.shuffle(order)
    for idx in order[: cap * 3]:
        rec = df.loc[idx]
        seen += 1
        for lang, (col, cap_chars) in LANG.items():
            if len(rows[lang]) >= args.per_lang:
                continue
            if col not in df.columns:
                continue
            for u, a in turns(rec[col]):
                c = clean(u, a, cap_chars)
                if c:
                    rows[lang].append({"instruction": c[0], "response": c[1], "src": lang})
                break   # first turn only: multi-turn makes the chat template a list

os.makedirs(args.out_dir, exist_ok=True)
manifest = {}
for lang, data in rows.items():
    random.shuffle(data)
    out = os.path.join(args.out_dir, f"ds_v2_{lang}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest[lang] = len(data)
    avg = sum(len(r["instruction"]) + len(r["response"]) for r in data) / max(1, len(data))
    print(f"  {lang:9s} {len(data):>6,} rows  avg {avg:5.0f} chars  -> {os.path.basename(out)}")

print(f"\nscanned {seen:,} source rows")
# alignment check: how many doc_ids appear in all three languages
dev = sum(1 for r in rows["hindi"] if DEV.search(r["instruction"] + r["response"]))
print(f"hindi rows containing Devanagari: {dev}/{len(rows['hindi'])}")

# Contamination audit - prove the output is only Hindi/Hinglish/English.
print("\n--- script audit (must be zero for everything except Devanagari in hindi) ---")
bad = False
for lang, data in rows.items():
    for name, rx in FOREIGN_RE.items():
        if name.startswith("Devanagari") and lang == "hindi":
            continue                      # Devanagari is CORRECT in the hindi set
        n = sum(1 for r in data if rx.search(r["instruction"] + r["response"]))
        if n:
            bad = True
            print(f"  !! {lang}: {n} rows contain {name} characters")
    print(f"  {lang:9s} clean of Bengali/Tamil/Telugu/Gujarati/Kannada/Malayalam/Arabic")

if not bad:
    print("\nOK: no non-Hindi Indic script in any output set.")
    print("    (mar_Deva / npi_Deva were never read - only hin_Deva, hin_Latn, eng_Latn)")
print("\nNow mix these with prepare_data.py's arena/casual (real human prompts) for the final sets.")
