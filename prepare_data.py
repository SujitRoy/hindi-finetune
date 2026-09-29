#!/usr/bin/env python3
"""
Build a Hinglish SFT mix for LFM2.5-1.2B-Instruct-Uncensored.

Stdlib only (csv + requests). Reads HF CSVs directly over HTTPS, no datasets/pyarrow.

  python3 prepare_data.py --out train.jsonl
"""
import argparse, csv, json, os, random, re, sys, time
import requests

random.seed(3407)

SOURCES = {
    # name: (repo, file, instr_col, resp_col, license, note)
    "arena": ("one-thing/chatbot_arena_conversations_hinglish", "data.csv",
              "user_hinglish", "assistant_hinglish", "apache-2.0",
              "real human prompts from chatbot arena, translated replies"),
    "casual": ("Abhishekcr448/Hinglish-Everyday-Conversations-1M", "hinglish_conversations.csv",
               "input", "output", "mit",
               "natural romanized everyday chat, very short"),
    "english": ("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
                "instruction", "response", "cc-by-sa-3.0",
                "anti-forgetting mix, single-file JSONL"),
}

# Devanagari Hindi from ai4bharat/indic-instruct-data-v0.1. Chosen because it ships
# per-row quality_metrics (chrF/sacreBLEU) - flan_v2/hi is badly translated and must be
# filtered, while oasst1/hi is real OpenAssistant conversation.
# wikihow/hi is deliberately absent: its parquet has a different schema
# (title/intro/steps) and no quality_metrics, so it cannot be chrF-filtered.
HINDI_CONFIGS = [("oasst1", 20000), ("dolly", 15000)]
HINDI_MIN_CHRF = 70
HINDI_REPO = "ai4bharat/indic-instruct-data-v0.1"
DS_FILE = {"arena": "ds_arena_hinglish.jsonl", "casual": "ds_casual_hinglish.jsonl",
           "hindi": "ds_hindi_devanagari.jsonl", "english": "ds_english_dolly.jsonl"}

# Devanagari costs 1.45 tok/char, so a Hindi row that reads fine still blows past
# MAX_SEQ_LENGTH. Measured on unfiltered data: median 1110 / p90 2529 tokens, 52.7%
# TRUNCATED - and a truncated target teaches the model never to finish its reply.
# 620 chars * 1.45 + ~15 tokens of template ~= 915, comfortably inside 1024.
HINDI_MAX_CHARS = 620


DEV = re.compile(r"[\u0900-\u097F]")          # any Devanagari codepoint
CODE_FENCE = re.compile(r"```")

# The translated-reply sources contain the ORIGINAL model refusing. Training on those
# teaches the model to refuse, which is the exact opposite of what this dataset is for.
REFUSAL = re.compile(
    r"\b(i'?m sorry|i am sorry|i apolog|i'?m sorry,|sorry, but|"
    r"i (?:can'?t|cannot|can not|won'?t|will not)\b|"
    r"i'?m (?:not able|unable|afraid)\b|i am (?:not able|unable)\b|"
    r"as an ai\b|as a language model\b|i must (?:decline|refuse)\b|"
    r"(?:can'?t|cannot|unable to) (?:assist|help|provide|comply)|"
    r"not (?:appropriate|necessary|able to be)|should not be used|"
    r"i do(?:n'?t| not) feel comfortable|i (?:don'?t|do not) (?:provide|engage))",
    re.I)


def _lines(url):
    """Stream a remote text file line by line so a 1M-row CSV never lands in RAM."""
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if raw:
                yield raw


def load_csv(repo, path, instr, resp, cap):
    """Read at most `cap` (instruction, response) pairs out of a remote CSV."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    rows, header = [], None
    for line in _lines(url):
        if header is None:
            header = next(csv.reader([line]))
            continue
        rec = dict(zip(header, next(csv.reader([line]))))
        i, o = (rec.get(instr) or "").strip(), (rec.get(resp) or "").strip()
        if i and o:
            rows.append((i, o))
            if len(rows) >= cap:
                break
    return rows


def load_english(n, workers=4):
    """dolly-15k ships as ONE .jsonl -> a single streamed GET. No paging, no rate limits."""
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    rows = []
    for line in _lines(url):
        d = json.loads(line)
        i, o = (d.get("instruction") or "").strip(), (d.get("response") or "").strip()
        ctx = (d.get("context") or "").strip()
        if ctx:                      # dolly splits prompt across instruction + context
            i = f"{i}\n\n{ctx}"
        if i and o:
            rows.append((i, o))
    print(f"    dolly-15k: {len(rows)} rows fetched")
    return rows[:n]


def load_hindi(caps=HINDI_CONFIGS, min_chrf=HINDI_MIN_CHRF, need=6000, cache="hi_cache"):
    """Devanagari instruction pairs, filtered on the dataset's own chrF score.

    Pulls the parquet shards straight off the HF CDN rather than the datasets-server
    rows API - the rows API hard-429s after a few hundred pages and each retry costs
    more than just downloading the (few-MB) shard.

    oasst1/hi -> messages[] (first user turn -> first assistant turn)
    dolly/hi  -> instruction + context + response
    """
    import io
    import pandas as pd

    os.makedirs(cache, exist_ok=True)
    rows = []
    # chrF and the char cap cull a lot, so over-fetch 4x before sampling down.
    target = need * 4
    for cfg, _cap in caps:
        if len(rows) >= target:
            break
        path = f"{cfg}/hi-00000-of-00001.parquet"
        local = os.path.join(cache, f"{cfg}_hi.parquet")
        if not os.path.exists(local):
            url = f"https://huggingface.co/datasets/{HINDI_REPO}/resolve/main/{path}"
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            with open(local, "wb") as f:
                f.write(r.content)
        df = pd.read_parquet(local)
        if "quality_metrics" not in df.columns:
            print(f"    {cfg}: skipped (no quality_metrics, cannot chrF-filter)", flush=True)
            continue
        n0 = len(df)
        df = df[df["quality_metrics"].map(lambda q: (q or {}).get("chrF", 0) >= min_chrf)]
        for rec in df.to_dict("records"):
            # pandas hands back numpy arrays for list columns, so `if rec[...]` raises
            # "truth value ambiguous". Every check must be `is not None`.
            msgs = rec.get("messages")
            if msgs is not None and len(msgs):
                u = next((x["content"] for x in msgs if x.get("role") == "user"), None)
                a = next((x["content"] for x in msgs if x.get("role") == "assistant"), None)
                if u and a:
                    rows.append((u.strip(), a.strip()))
            else:
                i = (rec.get("instruction") or "").strip()
                o = (rec.get("response") or "").strip()
                ctx = (rec.get("context") or "").strip()
                if ctx:
                    i = f"{i}\n\n{ctx}"
                if i and o:
                    rows.append((i, o))
            if len(rows) >= target:
                break
        print(f"    {cfg}: {n0:,} rows -> {len(df):,} pass chrF>={min_chrf} "
              f"-> {len(rows):,} kept", flush=True)
    return rows


def clean(i, o, allow_devanagari=False, max_chars=None):
    """Drop everything that would poison an SFT run for this tokenizer."""
    if not allow_devanagari and (DEV.search(i) or DEV.search(o)):
        return None                                   # Latin-only path
    if max_chars and (len(i) + len(o)) > max_chars:
        return None                                   # would truncate the target
    if CODE_FENCE.search(o) or CODE_FENCE.search(i):
        return None                                   # arena CSVs mangle code blocks
    if REFUSAL.search(o):
        return None                                   # would TRAIN the model to refuse
    if not (2 <= len(i.split()) <= 120):
        return None                                   # empty or runaway prompts
    if not (1 <= len(o.split()) <= 600):
        return None
    if len(i) > 1500 or len(o) > 4000:
        return None
    return (i, o)


TIERS = {
    # name         arena   casual  hindi  english
    "smoke":       (      5,       5,     5,      5),   # pipeline test, seconds
    "kaggle30":    (   4000,    1500,  3500,   1000),   # one-session bilingual mix
    "full":        (  11000,    6000, 12000,   5000),
    # one per 30-min Kaggle session, chained: each session trains on the PREVIOUS
    # session's pushed output, so knowledge accumulates across VM resets.
    "s1_hinglish": (   4000,    1500,     0,      0),
    "s2_hindi":    (      0,       0,  6000,      0),
    "s3_english":  (      0,       0,     0,   5000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="kaggle30",
                    help="kaggle30 = one-session mix; s1_*/s2_*/s3_* = chained sessions")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.n_arena, args.n_casual, args.n_hindi, args.n_english = TIERS[args.tier]
    os.makedirs(args.out_dir, exist_ok=True)

    out, stats, per_source = [], {}, {}

    def take(name, raw, n, why, fname):
        """Filter -> shuffle -> cap. Over-fetch 3x then sample so filtering still fills n."""
        kept = [c for c in (clean(*r) for r in raw) if c]
        random.shuffle(kept)
        sel = kept[:n]
        per_source[name] = sel
        out.extend({"instruction": i, "response": o, "src": name} for i, o in sel)
        stats[name] = (len(raw), len(sel))
        print(f"    {name:8s} scanned {len(raw):>7,}  kept {len(sel):>6,}  {why}")

    # --- 1. Arena: the instruction-following backbone -------------------------
    # Guarded: take() writes the per-source file, so calling it with n=0 would
    # clobber ds_arena_hinglish.jsonl with an empty file and lose 4,000 cached rows.
    if args.n_arena:
        repo, path, i_c, o_c, _, _ = SOURCES["arena"]
        take("arena", load_csv(repo, path, i_c, o_c, args.n_arena * 3 + 50),
             args.n_arena, "real prompts + substance", "ds_arena_hinglish.jsonl")

    # --- 2. Casual: natural register, short. Sampled, not flooded --------------
    if args.n_casual:
        repo, path, i_c, o_c, _, _ = SOURCES["casual"]
        take("casual", load_csv(repo, path, i_c, o_c, args.n_casual * 3 + 50),
             args.n_casual, "natural romanization", "ds_casual_hinglish.jsonl")

    # --- 3. Hindi (Devanagari): chrF-filtered on the dataset's own quality score --
    if args.n_hindi:
        raw = load_hindi(need=args.n_hindi)
        kept = [c for c in (clean(*r, allow_devanagari=True, max_chars=HINDI_MAX_CHARS)
                            for r in raw) if c]
        random.shuffle(kept)
        per_source["hindi"] = kept[:args.n_hindi]
        out.extend({"instruction": i, "response": o, "src": "hindi"}
                   for i, o in per_source["hindi"])
        stats["hindi"] = (len(raw), len(per_source["hindi"]))
        print(f"    {'hindi':8s} scanned {len(raw):>7,}  kept {len(per_source['hindi']):>6,}  Devanagari")

    # --- 4. English: prevents catastrophic forgetting of English + refusals ----
    if args.n_english:
        take("english", load_english(args.n_english + 50, workers=args.workers),
             args.n_english, "anti-forgetting", "ds_english_dolly.jsonl")

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    # per-source files, so re-weighting never needs a re-download
    for name, rows in per_source.items():
        dump(os.path.join(args.out_dir, DS_FILE[name]),
             [{"instruction": i, "response": o, "src": name} for i, o in rows])

    random.shuffle(out)
    mix_path = dump(os.path.join(args.out_dir, f"train_{args.tier}.jsonl"), out)

    print(f"\n{'source':10s} {'scanned':>10s} {'kept':>7s} {'share':>7s}   why")
    why = {"arena": "substance + real prompts", "casual": "natural romanization",
           "hindi": "Devanagari, chrF-filtered", "english": "anti-forgetting"}
    for k, (scanned, kept) in stats.items():
        print(f"{k:10s} {scanned:>10,} {kept:>7,} {100*kept/len(out):>6.1f}%   {why.get(k,'')}")
    print(f"\nmix: {mix_path}  ({len(out):,} rows)")
    print("per-source: " + ", ".join(sorted(
        f for f in os.listdir(args.out_dir) if f.startswith("ds_"))))


if __name__ == "__main__":
    main()
