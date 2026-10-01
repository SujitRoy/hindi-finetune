#!/usr/bin/env python3
"""Final gate on the generated dataset. Every number here is a rejection that
would otherwise reach training."""
import json, re, collections
rows = [json.loads(l) for l in open("/home/ubuntu/hindi-finetune/teach/teacher_gen.jsonl",
                                    encoding="utf-8")]
n = len(rows); print(f"rows {n:,}")
if not n: raise SystemExit
hi = [r for r in rows if r["lang"] == "hindi"]; hg = [r for r in rows if r["lang"] == "hinglish"]
print(f"  hindi {len(hi):,}  hinglish {len(hg):,}   ratio {len(hi)/max(1,n):.0%}/{len(hg)/max(1,n):.0%}")
for name, sub in (("hindi", hi), ("hinglish", hg)):
    if not sub: continue
    w = sorted(len(r["response"].split()) for r in sub)
    d = sorted(sum(1 for c in r["response"] if "ऀ" <= c <= "ॿ")/max(1, len(r["response"]))
               for r in sub)
    print(f"  {name:9s} words med {w[len(w)//2]} p10 {w[len(w)//10]} p90 {w[int(len(w)*.9)]} | "
          f"dev med {d[len(d)//2]:.2f} min {d[0]:.2f}")
dup = len(rows) - len({r["instruction"].strip().lower() for r in rows})
print(f"  duplicate instructions: {dup}")
END = re.compile(r"[।.?!\"'”’)\]]\s*$")
print(f"  truncated: {sum(1 for r in rows if not END.search(r['response']))}")
print(f"  leading whitespace: {sum(1 for r in rows if r['response'][:1].isspace())}")
REFU_HI = re.compile(r"(क्रृपया (संदर्भ|पाठ)|कोई (पाठ|संदर्भ) (नहीं|नही)|(नहीं|नही) (बता सकता|बता सकती)|मुझे (नहीं|नही) पता)")
print(f"  refusals: {sum(1 for r in rows if REFU_HI.search(r['response']))}")
print(f"  sources: {dict(collections.Counter(r.get('plang') for r in rows))}")
print("\n  --- samples ---")
for r in hi[:2] + hg[:2]:
    print(f"  [{r['lang']}] Q: {r['instruction'][:90]}")
    print(f"        A: {r['response'][:230]}")
