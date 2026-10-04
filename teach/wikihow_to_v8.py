#!/usr/bin/env python3
"""hi_cache/wikihow_hi.parquet -> wikihow_v8.jsonl (v8 fact/procedural rows).

Why this file exists: the v8 corpus had 52 verifiable-fact rows out of 66,502. Facts is
the register the model is worst at and the one the teacher cannot be trusted to write -
it invents dates. WikiHow's Hindi edition is human-written Hindi, not machine
translated, so it is the only real fact-ish source on this box.

Provenance check first, because it decided everything. hi_cache/dolly_hi.parquet looked
like 10,400 ready-made fact rows and was REJECTED: it carries backtranslated_instruction
/ backtranslated_response columns, i.e. it is English text run through a translator, and
the tell is visible - a question says एनिहिलेशन and its own answer says विनाश for the
same book. That is the `adaption` pool again, the one that put 1,245 Marathi rows into
v6. wikihow_hi is native: the Hindi is authored, and the romanised words in it
(फिगर स्केटर्स, आइस स्केटिंग) are how Hindi technical writing actually reads, which makes
them free Hinglish signal rather than noise.

Two data repairs, both measured rather than assumed:
  1. Text is percent-encoded ("%E0%A4%8F" -> ए). One urllib.parse.unquote handles the
     titles as measured; decode() loops up to 4 times anyway so a doubly-encoded field
     cannot sneak through, and the cap stops a literal '%' in prose from looping.
  2. The `messages` assistant turn is USELESS as shipped: it is the intro followed by
     27 numbered lines that each END IN A COLON and have no body - measured on the first
     1,500 docs, 24,433 of 24,433 numbered lines (100%) are bare headlines. The bodies
     are in the separate `steps` array (keys: description, number, picture, summary). So
     the answer is rebuilt as intro + "N. summary: description", which is what the
     website actually shows.

Self-check: --selfcheck renders the real chat template and runs the real unsloth_zoo
masker over the output, same contract as teach/oasst_to_v8.py.
"""
import os, re, sys, json, argparse, collections, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "hi_cache", "wikihow_hi.parquet")
OUT = os.path.join(HERE, "wikihow_v8.jsonl")
TOK_DIR = os.path.join(ROOT, "tokenizer-hi")
INSTRUCTION_PART = "user\n"
RESPONSE_PART = "assistant\n"
MAX_SEQ_LENGTH = 4096
MIN_ANSWER_WORDS = 30      # a 2-sentence stub teaches nothing
MIN_STEPS_WITH_BODY = 3    # below this the rebuilt answer is mostly the intro

# Same calibration as audit_v8.py / the notebook cell 9 gate - Marathi markers, both
# sides anchored so Delhi (मला inside दिल्ली) cannot fire.
MAR = re.compile(r"(?<![\u0900-\u097f])(आहे|आणि|तुम्ही|तुम्हाला|पण|मला|माझे|माझा|माझी|"
                 r"झाले|झाला|काय|कोणता|कोणती|म्हणजे|मुळे|याच|दिला|दिली|सकतो|असतो|आहोत)"
                 r"(?![\u0900-\u097f])")
HIN = re.compile(r"(?<![\u0900-\u097f])(है|हैं|हूँ|और|क्या|में|से|के|लिए|नहीं|हुआ|होता|"
                 r"करना|किया|गया|मिला|दिया|सकता|सकती|रहा|रही)(?![\u0900-\u097f])")
FOREIGN = re.compile("[\u0980-\u09ff\u0a80-\u0aff\u0b80-\u0dff\u3040-\u30ff"
                     "\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff]")
DEV = re.compile(r"[\u0900-\u097f]")
PCT = re.compile(r"%[0-9A-Fa-f]{2}")


def decode(s, limit=4):
    """Unquote repeatedly. wikihow_hi stores text double-encoded, so one pass is not
    enough; the cap is so a real '%' in prose cannot make this quadratic."""
    s = str(s or "")
    for _ in range(limit):
        if not PCT.search(s):
            return s
        out = urllib.parse.unquote(s)
        if out == s:
            return s
        s = out
    return s


def marathi(text):
    return len(set(MAR.findall(text))) > len(set(HIN.findall(text)))


def devshare(text):
    n = len(text) or 1
    return len(DEV.findall(text)) / n


def build_question(row):
    msg = row.get("messages")
    if msg is not None and len(msg):
        q = decode(msg[0]["content"]).strip()
        if q:
            return q
    t = decode(row.get("title")).strip()
    return f"कैसे {t}?" if t and not t.startswith("कैसे") else t


def build_answer(row, tok, budget):
    """intro + as many step bodies as fit `budget` tokens, cut at a step boundary."""
    intro = decode(row.get("intro")).strip()
    steps = row.get("steps")
    if steps is None or not intro:
        return None
    parts, used = [intro], ntok(tok, intro)
    if used > budget:
        return None
    for st in steps:
        if not isinstance(st, dict):
            continue
        body = decode(st.get("description")).strip()
        if not body:
            continue
        head = decode(st.get("summary")).strip().rstrip(":")
        n = st.get("number")
        num = f"{int(n)}. " if isinstance(n, (int, float)) and n else ""
        chunk = f"{num}{head}: {body}" if head else f"{num}{body}"
        c = ntok(tok, chunk)
        if used + c + 2 > budget:
            break
        parts.append(chunk)
        used += c + 2
    # The question asks for steps ("सारांश चरणों में प्रतिक्रिया"), so an answer that is
    # only the intro does not answer it. One body minimum; the doc is dropped rather than
    # truncated if its first step alone does not fit the budget.
    if len(parts) - 1 < 1:
        return None
    return "\n\n".join(parts).strip()


def ntok(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


GLOSS = re.compile(r"\((([A-Za-z][A-Za-z0-9'\u2019 .\-]{2,30}))\)")


def clean(row, tok, budget):
    """-> (row_or_None, drop_reason)"""
    q, a = build_question(row), build_answer(row, tok, budget)
    if not q or not a:
        return None, "no_text"
    if len(a.split()) < MIN_ANSWER_WORDS:
        return None, "too_short"
    if marathi(q + " " + a):
        return None, "marathi"
    if devshare(a) < 0.30:
        return None, "not_devanagari"
    if FOREIGN.search(q) or FOREIGN.search(a):
        return None, "foreign_script"
    if re.search(r"%[0-9A-Fa-f]{2}", q + a):
        return None, "still_encoded"
    return {"instruction": q, "response": a, "lang": "hindi", "plang": "hindi",
            "src": "wikihow_hi", "reg": "fact", "topic": "howto",
            "budget": "long", "judge": 4}, None


def n_tokens(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def render(tok, r):
    return tok.apply_chat_template(
        [{"role": "user", "content": r["instruction"]},
         {"role": "assistant", "content": r["response"]}],
        tokenize=False, add_generation_prompt=False).removeprefix(tok.bos_token or "")


def convert(budget=1600):
    import pandas as pd   # python3.13 on this box
    from transformers import AutoTokenizer
    df = pd.read_parquet(SRC)
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    drop = collections.Counter()
    out, seen = [], set()
    for _, row in df.iterrows():
        r, why = clean(row.to_dict(), tok, args.budget)
        if r is None:
            drop[why] += 1
            continue
        k = r["instruction"].strip().lower()
        if k in seen:
            drop["dupe"] += 1
            continue
        seen.add(k)
        r["tokens"] = n_tokens(tok, render(tok, r))
        if r["tokens"] > MAX_SEQ_LENGTH:
            drop["over_cap"] += 1
            continue
        out.append(r)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    w = sorted(len(r["response"].split()) for r in out)
    t = sorted(r["tokens"] for r in out)
    p = lambda q: t[int(len(t) * q)] if t else 0
    print(f"wrote {OUT}: {len(out):,} of {len(df):,} rows")
    for k, v in drop.most_common():
        print(f"  drop {k:18} {v:,}")
    if out:
        print(f"  answer words p10/med/p90 = {w[int(len(w)*.1)]}/{w[len(w)//2]}/{w[int(len(w)*.9)]}")
        print(f"  real tokens p50/p95/max = {p(.5):,}/{p(.95):,}/{t[-1]:,} (cap {MAX_SEQ_LENGTH:,})")
        print(f"  corpus tokens = {sum(t)/1e6:.1f}M")
    return out


def selfcheck(rows, limit=None):
    from transformers import AutoTokenizer
    sys.path.insert(0, "/tmp/ulz/x")
    os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from unsloth_zoo.dataset_utils import train_on_responses_only as zoo
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    fn = zoo(None, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
             tokenizer=tok, return_function=True)
    rws = rows[:limit] if limit else rows
    over = bad = 0
    for r in rws:
        ids = tok(render(tok, r), add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_SEQ_LENGTH:
            over += 1
            continue
        lab = fn({"input_ids": [ids]})["labels"][0]
        trained = sum(1 for x in lab if x != -100)
        if trained == 0 or trained == len(ids):
            bad += 1
    print(f"selfcheck {len(rws):,} rows: over_cap={over} bad_masking={bad} "
          f"-> {'PASS' if not (over or bad) else 'FAIL'}")
    return not (over or bad)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--selfcheck", action="store_true")
    a.add_argument("--limit", type=int, default=400)
    a.add_argument("--budget", type=int, default=int(os.environ.get("WH_BUDGET", 1600)))
    args = a.parse_args()
    rows = convert(args.budget)
    if args.selfcheck:
        sys.exit(0 if selfcheck(rows, args.limit) else 1)
