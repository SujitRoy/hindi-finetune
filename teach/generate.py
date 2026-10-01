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

CFG     = "/home/ubuntu/.pi/agent/models.json"
PROMPTS = "/home/ubuntu/hindi-finetune/teach/prompts.jsonl"
OUT     = "/home/ubuntu/hindi-finetune/teach/teacher_gen.jsonl"
MODEL   = "stealth/space-bunny-alpha"   # openrouter, cost 0/0, reasoning false
TARGET  = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CONC    = int(sys.argv[2]) if len(sys.argv) > 2 else 8   # 8, not 16: the Hindi/Hinglish run is already using 16
TIMEOUT = 180           # one request. Mercury is a reasoning model; 383 tokens of
                        # reasoning was not enough and content came back null.
RETRIES = 3

_pr = "openrouter"
_p = json.load(open(CFG))["providers"][_pr]
BASE, KEY = _p["baseUrl"], _p["apiKey"]

DEV_SYS = ("Aap ek helpful assistant ho. User ke sawaal ka jawab POORI tarah se, saaf aur natural "
           "Devanagari Hindi mein likho. English se seedha translation mat karo - waise likho jaise "
           "ek native Hindi speaker likhta hai. Markdown ya bullet points use mat karo, plain "
           "paragraph likho. Jawab 30 se 50 shabd ka likho. Isse zyada lamba mat likho.")
HING_SYS = ("Aap ek helpful assistant ho. User Hinglish mein likhta hai (romanized Hindi + English "
            "mix). Aap bhi isi tarah romanized Hinglish mein jawab do. Poora aur sahi jawab do. "
            "Varanak ke liye 'hai' aur 'hain' ka sahi prayog karo, 'h' ya 'kr' jaise shortcut mat "
            "likho. Markdown ya bullets use mat karo. Jawab 30 se 50 shabd ka likho.")
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

def accept(r, want_dev, instr=None, tgt=None):
    w = len(r.split())
    if w < 30:                           return False, "short"
    if w > 160:                          return False, "ramble"
    if not ENDS.search(r):               return False, "truncated"
    if BAD.search(r):                    return False, "shorthand"
    if REFU.search(r) or REFU_HI.search(r):  return False, "refusal"
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

def call(sys_p, user):
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": sys_p},
        {"role": "user",   "content": user}],
        "temperature": 0.7, "max_tokens": 1500}).encode()
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
# interleave targets so Hindi and Hinglish both fill up even if we stop early
jobs = []
for p in todo:
    jobs.append((p["instruction"], p["plang"]))
random.shuffle(jobs)

lock = threading.Lock()
kept = Counter(); drop = Counter(); fh = open(OUT, "a", encoding="utf-8")
t0 = time.time(); ntok = 0

EN_SYS = ("You are a helpful assistant. Answer the user's question directly, "
          "completely and in plain English. Use natural, well-formed English prose. "
          "Do NOT translate the question into another language and do not answer in "
          "Hindi unless the user explicitly asks for Hindi. Use markdown or bullets "
          "only if they genuinely help. Write 30 to 50 words.")

def work(job):
    instr, plang = job
    # Language MIRRORING, not unconditional Devanagari. The first dataset was
    # 100% non-English output - 810 Hindi + 369 Hinglish + 0 English - with
    # English prompts answered in Hindi. The model learned "always output
    # Devanagari" and now answers English questions in Hindi. `en` prompts must
    # produce English, or there is no signal at all for that mapping.
    # Language MIRRORING. Strictly one target language per prompt language:
    #   en / en_en  -> English answer
    #   hi          -> Devanagari answer
    #   hinglish    -> romanized Hinglish answer
    #
    # `en` used to be grouped with `hi`, so every English prompt was answered in
    # Devanagari. That produced 2,153 English-question -> Hindi-answer rows and
    # zero English -> English, and the model learned "always output Devanagari":
    # it answered English questions in Hindi. The routing has to be taught
    # explicitly or not at all.
    if plang in ("en", "en_en"):
        want_dev, sys_p, tgt = False, EN_SYS, "english"
    elif plang == "hi":
        want_dev, sys_p, tgt = True, DEV_SYS, "hindi"
    else:
        want_dev, sys_p, tgt = False, HING_SYS, "hinglish"
    try:
        a = call(sys_p, instr)
    except Exception as e:
        return ("error", f"{type(e).__name__}")
    ok, why = accept(a, want_dev, instr, tgt)
    if not ok: return ("drop", why)
    return ("keep", {"instruction": instr, "response": a, "src": "teacher",
                     "plang": plang, "lang": tgt})

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
