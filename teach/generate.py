#!/usr/bin/env python3
"""Teacher generation with Mercury-2.5, run on this box.

Runs here rather than on Kaggle for two reasons. The InceptionLabs key lives in
~/.pi/agent/models.json and never leaves this machine, and the calls are I/O
bound, so no GPU is needed and the 12-hour Kaggle session wall does not apply.

  throughput  11,134 rows/h at concurrency 16 (measured)
  cost        $0.15/M output tokens, ~2,100 tokens per answer of which ~100 is
              the answer because the model is a reasoning model
              8,000 rows ~ $2.50

Resumable by prompt hash, so a crash costs at most one batch.
"""
import json, os, re, sys, time, random, threading, urllib.request, urllib.error
import concurrent.futures as cf
from collections import Counter

# baseUrl/apiKey/model come from teach/teacher.local.json (gitignored, mode 600).
# This repo is public and the teacher is a private preview on someone else's plan.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_cfg import TEACHER
PROMPTS = os.environ.get("PROMPTS", "/home/ubuntu/hindi-finetune/teach/prompts_v8.jsonl")
OUT     = os.environ.get("GEN_OUT", "/home/ubuntu/hindi-finetune/teach/teacher_gen_v8.jsonl")
TARGET  = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
CONC    = int(sys.argv[2]) if len(sys.argv) > 2 else 16   # measured 11k rows/h at 16
TIMEOUT = 180           # one request. Mercury is a reasoning model; 383 tokens of
                        # reasoning was not enough and content came back null.
RETRIES = 3

BASE, KEY, MODEL = TEACHER["baseUrl"], TEACHER["apiKey"], TEACHER["model"]

# ---------------------------------------------------------------- length budgets
# v7 had ZERO rows under 15 words because the system prompt said "30 se 50 shabd" AND
# accept() demanded >=30. Median 47, every answer the same shape: the model has no 8-word
# greeting in 57,687 rows, so "kaise ho bhai" gets a 47-word topical dodge. Budgets are
# sampled per row and accept() bounds FOLLOW the sample, so the corpus gets a distribution.
# (ask_lo, ask_hi, keep_min, keep_max)
BUDGETS = {"short":  (8,  22,  4,  45),
           "medium": (28, 55, 20,  90),
           "long":   (60, 110, 45, 170)}
BUDGET_W = {"topical": (0.30, 0.45, 0.25), "chat": (0.85, 0.15, 0.00)}

def sys_for(tgt, lo, hi):
    # "do not come back only asking for more information" - the teacher's most common
    # failure under a short budget, and the exact behaviour that made v7 unusable for chat.
    # One anti-dodge instruction per target, in that target's own script.
    NO_DODGE = {
      "hindi": (" अगर प्रश्न थोड़ा अधूरा हो तब भी जो सबसे संभावित और उपयोगी उत्तर है वही दो; "
                "केवल और जानकारी माँगने भर में उत्तर खत्म मत करो।"),
      "hinglish": (" Agar sawaal thoda adhoora ho tab bhi jo sabse sambhavit aur useful jawab "
                   "hai wahi do; sirf aur information maangne mein jawab khatam mat karo."),
      "english": (" If the question is slightly incomplete, still give the most likely useful "
                  "answer; do not reply only by asking for more information."),
    }[tgt]
    if tgt == "hindi":
        return ("Aap ek helpful assistant ho. User ke sawaal ka jawab POORI tarah se, saaf aur "
                "natural Devanagari Hindi mein likho. English se seedha translation mat karo - "
                "waise likho jaise ek native Hindi speaker likhta hai. Markdown ya bullet points "
                f"use mat karo, plain paragraph likho. Jawab {lo} se {hi} shabd ka likho; na usse "
                "chhota, na usse bada." + NO_DODGE)
    if tgt == "hinglish":
        return ("Aap ek helpful assistant ho. User Hinglish mein likhta hai (romanized Hindi + "
                "English mix). Aap bhi isi tarah romanized Hinglish mein jawab do. Poora aur sahi "
                "jawab do. Varanak ke liye 'hai' aur 'hain' ka sahi prayog karo, 'h' ya 'kr' jaise "
                f"shortcut mat likho. Markdown ya bullets use mat karo. Jawab {lo} se {hi} word ka "
                "likho; na usse chhota, na usse bada." + NO_DODGE)
    return ("You are a helpful assistant. Answer directly in natural, well-formed English prose. "
            f"Write {lo} to {hi} words - no more and no less. No markdown unless asked."
            + NO_DODGE)

# The judge is the same model as the generator, so it grades its own evasive style leniently:
# in the first smoke run "Apni society ka naam, city, locality aur pin code batayein..."
# scored 4/5 and was kept. Directness is therefore enforced mechanically as well as judged.
DODGE = re.compile(r"(ka naam[^.]{0,40}batayein|pin ?code[^.]{0,30}batayein|"
                   r"locality[^.]{0,30}batayein|sandarb[^.]{0,30}batayein|"
                   r"context[^.]{0,40}(batayein|bata de|share karein)|"
                   r"apni baat[^.]{0,40}share karein|krpaya[^.]{0,30}(bata|doher)[^.]{0,20}batayein)", re.I)

def dodges(text):
    """True when the answer opens by demanding more input instead of answering."""
    return bool(DODGE.search(" ".join(text.split()[:18])))

BAD  = re.compile(r"(ai assistant|ai model|as an ai|मैं एक एआई|\bh\b(?!\w)|\bkr\b|\bkro\b)", re.I)
REFU = re.compile(r"(i am sorry|i cannot|i can't|maine pucha|mujhe nahi pata|"
                   r"i'm not able|as an ai)", re.I)
# Hindi refusals that the English REFU pattern does not catch. Measured: 3 of the
# first 93 rows were "आपने कोई पाठ प्रदान नहीं किया है, इसलिए मैं यह नहीं बता सकता" and
# survived. Training on those teaches the student to refuse.
REFU_HI = re.compile(r"(क्रृपया (संदर्भ|पाठ|सन्दर्भ)|कोई (पाठ|संदर्भ|सन्दर्भ) (नहीं|नही)|"
                      r"(नहीं|नही) (बता सकता|बता सकती)|मुझे (नहीं|नही) पता|"
                      r"मुझे (नहीं|नही) मालूम|यह (नहीं|नही) बता सकता)", re.I)
# A reply that shares almost no content word with the prompt is a non-sequitur.
# 83% of smangrul's Hindi bucket were non-sequiturs and no length filter catches it.
WORD = re.compile(r"[\u0900-\u097F]{3,}|[a-zA-Z]{4,}")
ENDS = re.compile(r"[।.?!\"'”’)\]]\s*$")   # the ASCII period matters: Hinglish ends in '.'

def accept(r, want_dev, instr=None, tgt=None, budget="medium"):
    _a, _b, minw, maxw = BUDGETS[budget]
    # dodge FIRST: a 16-word evasive answer would otherwise be reported as "below-budget",
    # which hides the real defect in the drop counters that audit the corpus.
    if dodges(r):                            return False, "dodge"
    if not ENDS.search(r):               return False, "truncated"
    if BAD.search(r):                    return False, "shorthand"
    if REFU.search(r) or REFU_HI.search(r):  return False, "refusal"
    w = len(r.split())
    if w < minw:                           return False, "below-budget"
    if w > maxw:                           return False, "above-budget"
    if not ENDS.search(r):               return False, "truncated"
    if BAD.search(r):                    return False, "shorthand"
    if REFU.search(r) or REFU_HI.search(r):  return False, "refusal"
    if dodges(r):                            return False, "dodge"
    d = sum("ऀ" <= c <= "ॿ" for c in r)/max(1, len(r))
    # Non-sequitur guard, same shape as the smangrul failure (83% of its Hindi
    # bucket). Only meaningful when both sides are in the SAME script - an English
    # prompt scored against a Devanagari answer is always 0.00 by construction.
    if instr:
        pi = sum(1 for c in instr if "\u0900" <= c <= "\u097F") / max(1, len(instr))
        # BOTH sides same script. "or" was wrong: every Devanagari answer made the
        # check fire even against an English prompt, where it always scores 0.00.
        if (pi > 0.35) == (d > 0.35):
            cw = {x.lower() for x in WORD.findall(instr)}
            rw = {x.lower() for x in WORD.findall(r)}
            if cw and rw and len(cw & rw) / len(cw) < 0.08:
                return False, "unrelated"
    t = r.split()
    tri = [tuple(t[i:i+3]) for i in range(len(t)-2)]
    if tri and len(set(tri))/len(tri) < 0.92: return False, "repetitive"
    # One ceiling per target language. English is 0.10, not the Hinglish 0.35:
    # at 0.35 a genuinely Hindi answer passes an English target.
    # Mirror the prompt's script too. A Devanagari question answered in romanized
    # Hinglish teaches the model that Devanagari input does not guarantee a
    # Devanagari answer - the same class of bug as English->Hindi, in reverse.
    if instr and tgt in ("hindi", "hinglish"):
        pi = sum(1 for c in instr if "\u0900" <= c <= "\u097F") / max(1, len(instr))
        if pi > 0.35 and d < 0.35:
            return False, "romanized_answer_to_devanagari"
    if tgt == "english"  and d > 0.10:   return False, "answer_in_hindi"
    if tgt == "hinglish" and d > 0.35:   return False, "went_devanagari"
    if tgt == "hindi"    and d < 0.55:   return False, "not_devanagari"
    return True, ""

def call(sys_p, user, max_tokens=1500, temp=0.7):
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": sys_p},
        {"role": "user",   "content": user}],
        "temperature": temp, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    last = None
    for a in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                j = json.loads(resp.read())
            c = j["choices"][0]["message"]["content"]
            if not c:
                last = RuntimeError(f"null content, {j.get('usage',{}).get('completion_tokens')} tok")
                time.sleep(2 * (a + 1)); continue          # reasoning ate the budget
            return c.strip()
        except Exception as e:
            last = e; time.sleep(2 * (a + 1))
    raise last

pool = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]
done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
print(f"pool {len(pool):,}   already have {len(done):,}   target {TARGET:,}   conc {CONC}", flush=True)

todo = [p for p in pool if p["key"] not in done]
# interleave targets so Hindi and Hinglish both fill up even if we stop early. The length
# budget is drawn here, seeded by key, so a resume reproduces the same assignment.
jobs = []
for p in todo:
    reg = p.get("reg", "topical")
    budget = random.Random(p["key"]).choices(list(BUDGETS), weights=BUDGET_W[reg], k=1)[0]
    jobs.append((p["instruction"], p["plang"], budget, p["key"], reg, p.get("topic", "")))
random.shuffle(jobs)

lock = threading.Lock()
kept = Counter(); drop = Counter(); fh = open(OUT, "a", encoding="utf-8")
t0 = time.time(); ntok = 0

# ---------------------------------------------------------------- self-judge pass
# The teacher is free, so use it twice: generate, then grade. v7 trained 57,687 rows whose
# grammaticality was never scored and later shipped 2,482 rows that teach evasive boilerplate.
# Same-model grading is biased, but a zero-cost gate that drops broken or dodging answers
# beats no gate; audit_v8.py tracks the pass rate so drift is visible.
J_SYS = ("You are a strict evaluator of Hindi and Hinglish assistant answers. Judge the ANSWER "
         "against the QUESTION on two things only: (1) grammar, spelling and natural word "
         "choice; (2) whether it answers directly instead of dodging, stalling or padding. "
         "5 = native-quality and answers the question. 4 = good with minor stiffness. "
         "3 = understandable but awkward or partly dodging. 1 = broken grammar or non-answer. "
         "Reply with exactly one digit 1-5 and nothing else.")

def judge(instr, ans):
    try:
        out = call(J_SYS, f"Question:\n{instr}\n\nAnswer:\n{ans}", max_tokens=300, temp=0.0)
    except Exception:
        return 0
    m = re.search(r"[1-5]", out)
    return int(m.group()) if m else 0

BUDGET_TOK = {"short": 300, "medium": 900, "long": 1500}

def work(job):
    instr, plang, budget, key, reg, topic = job
    # Language MIRRORING, strictly one target per prompt language:
    #   en / en_en -> English,  hi -> Devanagari,  else -> romanized Hinglish.
    # (An earlier revision grouped `en` with `hi`, which taught "always output Devanagari"
    # and made the model answer English questions in Hindi.)
    if plang in ("en", "en_en"):
        want_dev, tgt = False, "english"
    elif plang == "hi":
        want_dev, tgt = True, "hindi"
    else:
        want_dev, tgt = False, "hinglish"
    lo, hi = BUDGETS[budget][0], BUDGETS[budget][1]
    try:
        a = call(sys_for(tgt, lo, hi), instr, max_tokens=BUDGET_TOK[budget])
    except Exception as e:
        return ("error", f"{type(e).__name__}")
    # instr=None skips the non-sequitur guard: a chat reply legitimately shares almost no
    # content words with "kaise ho bhai", and the guard would delete every short row.
    ok, why = accept(a, want_dev, None if reg == "chat" else instr, tgt, budget)
    if not ok: return ("drop", why)
    sc = judge(instr, a)
    if sc < 4: return ("drop", f"judge{sc}")
    return ("keep", {"key": key, "instruction": instr, "response": a, "src": "teacher",
                     "plang": plang, "lang": tgt, "reg": reg, "topic": topic,
                     "budget": budget, "judge": sc})

def consume(r):
    global ntok
    with lock:
        if r[0] == "keep":
            fh.write(json.dumps(r[1], ensure_ascii=False) + "\n"); fh.flush()
            kept[r[1]["lang"]] += 1
        else:
            drop[r[1]] += 1
        n = sum(kept.values())
        if n and n % 100 == 0:
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] {n:,} kept  {el/60:.1f}m  "
                  f"{el/n:.2f}s/row  {n/el*3600:,.0f}/h  keep={dict(kept)} "
                  f"drop={dict(drop.most_common(6))}", flush=True)
            if n >= TARGET: raise KeyboardInterrupt

try:
    with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
        for r in ex.map(work, jobs[: TARGET * 3]):
            consume(r)
except KeyboardInterrupt:
    pass
finally:
    fh.close()
el = time.time() - t0
tot = sum(kept.values())
print(f"\nkept    : {dict(kept)}  total {tot:,}")
print(f"dropped : {dict(drop.most_common(12))}")
print(f"elapsed : {el/60:.1f} min   {el/max(1,tot):.2f} s/row   {tot/el*3600:,.0f} rows/h")
print(f"file    : {OUT}")
