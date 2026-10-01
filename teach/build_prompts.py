#!/usr/bin/env python3
"""Assemble the prompt pool for teacher generation.

The teacher only has to be good at RESPONSES, so the prompts just need to be
standalone, varied, and in the right language. Three sources:

  dolly      14,560 human-written English instructions. Used for both Hindi and
             Hinglish targets so the model also learns English question ->
             Hindi answer.
  oasst1_hi  real questions asked by real Hindi-speaking users. 39,518 turns
             reduce to 13,262 unique, but most are follow-ups referencing an
             earlier turn ("आपके जवाब के लिए धन्यवाद! क्या मैं उस समय...").
             Those are dead as standalone training prompts, so back-referential
             openers are dropped.
  selfinst   1,018 Hinglish questions, the only romanized-Hinglish prompt
             source available locally.
"""
import json, re, glob, os, random, hashlib

OUT = "/home/ubuntu/hindi-finetune/teach/prompts.jsonl"
random.seed(11)

DEV = lambda s: sum(1 for c in s if "ऀ" <= c <= "ॿ") / max(1, len(s))
BACKREF = re.compile(r"(आपके जवाब|आपने कहा|आपने बताया|उस समय|जैसा आपने|"
                     r"इसका मतलब|पिछले प्रश्न|ऊपर बताया|आपके पहले|"
                     r"आपके अनुसार|क्या आपने|जैसा कि|और इसके|उसी के|"
                     r"आपका पहला|पहले आप)", re.I)
EN_BACKREF = re.compile(r"\b(you (said|told|mentioned)|your (answer|response)|"
                        r"as you|that you)\b", re.I)

def clean(u):
    u = u.strip()
    if u.startswith("Translate"): return None
    return u or None

prompts = []   # (text, lang, src)

# --- dolly, English
for p in ["/kaggle/working/data_cache/dolly.jsonl", "/tmp/gt/dolly.jsonl"]:
    if os.path.exists(p): break
else:
    import urllib.request
    p = "/tmp/gt/dolly.jsonl"
    os.makedirs("/tmp/gt", exist_ok=True)
    with urllib.request.urlopen("https://huggingface.co/datasets/databricks/"
        "databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl", timeout=900) as r, \
         open(p, "wb") as f:
        f.write(r.read())
n_dolly = 0
for line in open(p, encoding="utf-8"):
    try: q = json.loads(line)["instruction"].strip()
    except Exception: continue
    if not q: continue
    w = len(q.split())
    if not (3 <= w <= 40): continue
    if q.lower().rstrip(" ?.").split()[-1] in ("or","and","is","are","the","a","an"): continue
    if re.match(r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b",
                q, re.I) and w <= 6: continue
    if EN_BACKREF.search(q): continue
    prompts.append((q, "en", "dolly")); n_dolly += 1

# --- oasst1 hi, standalone only
n_hi = 0
try:
    import pandas as pd
    df = pd.read_parquet("/home/ubuntu/hindi-finetune/hi_cache/oasst1_hi.parquet")
    seen = set()
    for m in df["messages"]:
        for x in list(m):
            if x.get("role") != "user": continue
            u = (x.get("content") or "").strip()
            if not (5 <= len(u.split()) <= 45): continue
            if BACKREF.search(u): continue
            if DEV(u) < 0.7: continue
            if "\n" in u: continue           # multi-turn fragments
            if u in seen: continue
            seen.add(u)
            prompts.append((u, "hi", "oasst1")); n_hi += 1
except Exception as e:
    print(f"oasst1 skipped: {e}")

# --- selfinst, Hinglish
n_hin = 0
try:
    import pandas as pd
    df = pd.read_parquet("/tmp/langcheck/selfinst.parquet")
    for _, r in df.iterrows():
        m = list(r["messages"])
        if len(m) < 2: continue
        u = m[0]["content"].strip()
        if not (4 <= len(u.split()) <= 45): continue
        if DEV(u) > 0.35: continue          # must be romanized
        prompts.append((u, "hinglish_p", "selfinst")); n_hin += 1
except Exception as e:
    print(f"selfinst skipped: {e}")

random.shuffle(prompts)
with open(OUT, "w", encoding="utf-8") as f:
    for t, lang, src in prompts:
        f.write(json.dumps({"instruction": t, "plang": lang, "src": src,
                            "key": hashlib.sha1(f"{lang}|{t}".encode()).hexdigest()[:16]},
                           ensure_ascii=False) + "\n")
print(f"wrote {len(prompts):,} prompts -> {OUT}")
print(f"  dolly(en)   {n_dolly:,}")
print(f"  oasst1(hi)  {n_hi:,}   after dropping follow-ups")
print(f"  selfinst    {n_hin:,} (hinglish prompts)")
