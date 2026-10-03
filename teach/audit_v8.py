#!/usr/bin/env python3
"""audit_v8.py - prove the corpus has the shape v7 lacked. One pass, no API, no GPU.

  python3 teach/audit_v8.py [file.jsonl]      # default teach/teacher_gen_v8.jsonl

v7's numbers, measured after the fact, are the baseline this compares against:
  response words p10/median/p90 = 40/47/52, ZERO rows <= 15 words
  instruction words median 13, only 23 rows <= 6 words (0.04%)
  chat register 291 rows (0.50%), multi-turn 1 row, 2,482 evasive rows
Those came from three places all enforcing one spike: the system prompt said
"30 se 50 shabd", accept() demanded >= 30 words, build_release MIN_WORDS was 20.
"""
import json, re, sys, collections, statistics, os

TARGETS = {  # the v8 shape, from BUDGET_W in generate.py
    "short_share":  0.20,   # rows <= 15 words
    "chat_share":   0.10,   # reg == "chat"
    "max_topic_p":  0.35,   # no single topic may dominate
}

DEV   = re.compile(r"[\u0900-\u097f]")
WORD  = re.compile(r"[A-Za-z\u0900-\u097f']+")
DODGE = re.compile(r"(ka naam[^.]{0,40}batayein|pin ?code[^.]{0,30}batayein|"
                   r"locality[^.]{0,30}batayein|context[^.]{0,40}(batayein|share karein)|"
                   r"apni baat[^.]{0,40}share karein)", re.I)
MAR   = re.compile(r"(?<![\u0900-\u097f])(आहे|आणि|तुम्ही|तुम्हाला|झाले|म्हणजे|मुळे|सकतो|असतो)"
                   r"(?![\u0900-\u097f])")

def words(s): return len(WORD.findall(s))
def devshare(s):
    L = [c for c in s if c.isalpha() or "ऀ" <= c <= "ॿ"]
    return sum(1 for c in L if "ऀ" <= c <= "ॿ") / max(1, len(L))

def rows_of(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                try: out.append(json.loads(l))
                except json.JSONDecodeError: pass
    return out

def main(path):
    rows = rows_of(path)
    if not rows:
        sys.exit(f"no rows in {path}")
    rl = sorted(words(r["response"]) for r in rows)
    il = sorted(words(r.get("instruction", "")) for r in rows)
    p = lambda a, q: a[min(len(a) - 1, int(len(a) * q))]
    short = sum(1 for x in rl if x <= 15)
    chat  = sum(1 for r in rows if r.get("reg") == "chat")
    dodges = sum(1 for r in rows if DODGE.search(" ".join(r["response"].split()[:18])))
    marathi = sum(1 for r in rows if MAR.search(r["response"]))
    # script mirroring: Devanagari question must not get a romanized answer
    mirror = sum(1 for r in rows
                 if devshare(r.get("instruction", "")) > 0.35 and devshare(r["response"]) < 0.35)
    topics = collections.Counter(r.get("topic", "?") for r in topics_iter(rows))
    judges = collections.Counter(r.get("judge", 0) for r in rows)
    langs  = collections.Counter(r.get("lang", "?") for r in rows)

    print(f"file                {path}")
    print(f"rows                {len(rows):,}")
    print(f"langs               {dict(langs)}")
    print(f"judge scores        {dict(sorted(judges.items()))}")
    print()
    print(f"response words  p10/med/p90   {p(rl,.1)}/{p(rl,.5)}/{p(rl,.9)}   (v7: 40/47/52)")
    print(f"  <=15 words    {short:,} ({100*short/len(rows):.1f}%)   target >= "
          f"{100*TARGETS['short_share']:.0f}%   {'OK' if short/len(rows) >= TARGETS['short_share'] else 'LOW'}")
    print(f"  chat register {chat:,} ({100*chat/len(rows):.1f}%)   target >= "
          f"{100*TARGETS['chat_share']:.0f}%   {'OK' if chat/len(rows) >= TARGETS['chat_share'] else 'LOW'}")
    print(f"instruction words p10/med/p90 {p(il,.1)}/{p(il,.5)}/{p(il,.9)}   (v7: 10/13/18)")
    print(f"  <=6 words     {sum(1 for x in il if x <= 6):,} "
          f"({100*sum(1 for x in il if x<=6)/len(il):.1f}%)   v7 was 0.04%")
    print()
    print(f"topics              {len(topics)} distinct, largest "
          f"{topics.most_common(1)[0][1]/len(rows):.0%} (cap {TARGETS['max_topic_p']:.0%})")
    for t, n in topics.most_common(8):
        print(f"    {n:6,}  {t[:60]}")
    print()
    bad = [("marathi rows", marathi), ("dodge rows", dodges),
           ("script-mirror violations", mirror)]
    for name, n in bad:
        print(f"  {'FAIL' if n else 'ok':4}  {name}: {n:,}")
    fails = [n for _, n in bad if n] or \
            [x for x, ok in [("short_share", short/len(rows) >= TARGETS["short_share"]),
                             ("chat_share", chat/len(rows) >= TARGETS["chat_share"])] if not ok]
    print("\n" + ("AUDIT PASS" if not fails else f"AUDIT FAIL: {fails}"))
    return 0 if not fails else 1

def topics_iter(rows):
    for r in rows: yield r

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "teacher_gen_v8.jsonl")))
