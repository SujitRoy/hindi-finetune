#!/usr/bin/env python3
"""Merge the expanded prompts in, dedupe, and keep the two languages balanced.

Balance matters: English and Hindi prompts both produce Devanagari answers, so
the Hindi bucket is fed by plang in (en, hi) and the Hinglish bucket by plang
hinglish. Left to a plain shuffle, one side runs out of prompts long before the
other and generation stops with a lopsided dataset.
"""
import json, random, hashlib, collections, os
random.seed(53)
BASE = "/home/ubuntu/hindi-finetune/teach"
cur = [json.loads(l) for l in open(f"{BASE}/prompts.jsonl", encoding="utf-8")]
if os.path.exists(f"{BASE}/extra_prompts.jsonl"):
    cur += [json.loads(l) for l in open(f"{BASE}/extra_prompts.jsonl", encoding="utf-8")]
seen, pool = set(), []
for p in cur:
    k = p["instruction"].strip().lower()
    if not k or k in seen: continue
    seen.add(k)
    p["instruction"] = p["instruction"].strip()
    p["key"] = hashlib.sha1(f"{p['plang']}|{p['instruction']}".encode()).hexdigest()[:16]
    pool.append(p)
random.shuffle(pool)
with open(f"{BASE}/prompts.jsonl", "w", encoding="utf-8") as f:
    for p in pool: f.write(json.dumps(p, ensure_ascii=False) + "\n")
c = collections.Counter(p["plang"] for p in pool)
hin = c["hi"] + c["en"]
print(f"pool {len(pool):,}:  hi {c['hi']:,}  en {c['en']:,}  hinglish {c['hinglish']+c['hinglish_p']:,}")
print(f"  -> can yield up to {hin:,} Hindi rows and {c['hinglish']+c['hinglish_p']:,} Hinglish rows")
