#!/usr/bin/env python3
"""Language routing must be one-to-one: a prompt's language picks the answer's.

This is the check that catches the bug that made the first teacher run collapse
to a single output language. `en` was grouped with `hi` in work(), so 2,153 rows
taught English-question -> Hindi-answer and zero taught English -> English. The
model learned "always output Devanagari" and answered English questions in
Hindi. Counts cannot see that; these six cases can.
"""
import re, textwrap, sys, os
src = open(os.path.join(os.path.dirname(__file__), "generate.py")).read()
ns = {"re": re}
exec(textwrap.dedent(src[src.index("BAD  = re.compile"):src.index("def call(")]), ns)
accept = ns["accept"]

HI = ("भारत की राजधानी नई दिल्ली है। यह शहर उत्तर भारत के मैदानी भाग में स्थित है और यहाँ "
      "संसद तथा राष्ट्रपति भवन जैसे प्रमुख स्थल हैं। यह आर्थिक और राजनीतिक गतिविधियों "
      "का प्रमुख केंद्र माना जाता है और यहाँ बहुत से लोग रहते हैं।")
EN = ("Family time is the unhurried part of the day that a family spends together, "
      "talking, playing or eating rather than on screens. It builds trust between "
      "members, helps children feel secure, and costs nothing but setting aside time.")
HING = ("Laptop slow hone ke kuch common reasons hote hain: purane background apps, kam "
        "RAM, aur disk space khatam hona. Pehle unused apps band karein, phir disk clean "
        "karein, aur RAM upgrade karwa lein agar phir bhi slow rahe.")
HIQ = "भारत की राजधानी क्या है और उसकी विशेषताएँ क्या हैं?"
CASES = [
    ("Hindi Q -> Hindi A",            "hindi",    HIQ,              HI,   True),
    ("English Q -> English A",        "english",  "What is family time?", EN,   True),
    ("Hinglish Q -> Hinglish A",      "hinglish", "Laptop slow kya karun?", HING, True),
    ("English Q -> Hindi A",          "english",  "What is family time?", HI,   False),
    ("Devanagari Q -> romanized A",   "hinglish", HIQ,              HING, False),
    ("Hinglish Q -> Devanagari A",    "hinglish", "Laptop slow kya karun?", HI,   False),
]
bad = 0
for name, tgt, q, a, want in CASES:
    ok, why = accept(a, False, q, tgt)
    if ok != want: bad += 1
    print(f"  {'PASS' if ok == want else 'FAIL'}  {name:30s} "
          f"{'accepted' if ok else 'reject: ' + why}")
print("=" * 62)
print(f"  FAIL - {bad} routing case(s) wrong" if bad else "  PASS - routing is one-to-one")
sys.exit(1 if bad else 0)
