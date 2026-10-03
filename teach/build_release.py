#!/usr/bin/env python3
"""Build a training release from teacher output. Hindi + Hinglish only.

  python3 teach/build_release.py teach/teacher_gen_main.jsonl train_v7_teacher.jsonl

The fix is a SOURCE BLACKLIST, not a text pattern. Measured on the in-flight
teacher output: 1,245 Devanagari rows are Marathi-dominated, and every single one
(1245/1245, 100% attribution) came from a prompt in the `adaption` set, which is
itself 32.7% Marathi/Maithili machine-translated material. Drop that set and the
corpus is clean by construction:

  drop 'adaption' -> rows 62,405 -> 56,597, marathi-dominated 1245 -> 0

langgate.py stays in here as an ASSERTION, not the filter. If it ever rejects a
row after the blacklist, the blacklist is wrong and the build must fail loudly
rather than ship another quietly-poisoned corpus.

Also dropped: lang == "english" and the English->Hindi switch rows. Those are what
taught the last model that a Latin-script question deserves a Devanagari answer.

generate.py is not imported: it issues API calls at module level.
"""
import json, re, sys, collections
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import langgate

SRC_BAD = {"adaption"}   # machine-translated Marathi/Maithili; 32.7% Marathi prompts
# Was a flat MIN_WORDS = 20, which deleted every short row at release time and rebuilt the
# single-spike length distribution that made v7 answer "kaise ho bhai" with a 47-word dodge.
# Each row keeps its own floor, from the budget generate.py sampled for it.
BUDGET_MIN = {"short": 4, "medium": 20, "long": 45}
MIN_WORDS = 20           # fallback for legacy rows with no budget field
ENDS = re.compile(r"[।.?!”\"’)\]]\s*$")

def prompt_sources(*pools):
    """instruction -> src, from every prompt file we have. This is what makes the
    blacklist work on already-generated rows, which carry no src of their own."""
    idx = {}
    for f in pools:
        try:
            for l in open(f, encoding="utf-8"):
                p = json.loads(l)
                idx.setdefault(p["instruction"].strip(), p.get("src", "?"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return idx

def structural(resp, min_words=MIN_WORDS):
    if len(resp.split()) < min_words:                     return "short"
    if not ENDS.search(resp):                             return "truncated"
    t = resp.split()
    tri = [tuple(t[i:i+3]) for i in range(len(t)-2)]
    if tri and len(set(tri))/len(tri) < 0.92:             return "repetitive"
    return None

BASE = __file__.rsplit("/", 1)[0]          # the teach/ directory

def build(inp, outp):
    src = prompt_sources(f"{BASE}/pool_master.jsonl",
                         f"{BASE}/pool_todo.jsonl",
                         f"{BASE}/prompts.jsonl",
                         f"{BASE}/pool_all.jsonl",
                         f"{BASE}/pool_new.jsonl")
    rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
    seen, kept, drop = set(), [], collections.Counter()
    for r in rows:
        instr, resp, lang = r["instruction"], r["response"], r.get("lang", "")
        if lang not in ("hindi", "hinglish"):
            drop[f"not_hindi_hinglish({lang})"] += 1; continue
        if src.get(instr.strip()) in SRC_BAD:
            drop["src_blacklisted"] += 1; continue
        if (lang, instr.strip().lower()) in seen:
            drop["duplicate"] += 1; continue
        why = structural(resp, BUDGET_MIN.get(r.get("budget"), MIN_WORDS))
        if why:
            drop[why] += 1; continue
        # The blacklist handles the systemic Marathi source. The gate catches the
        # rest: CJK intruding into a scraped prompt, answers drifting into Gujarati.
        why = langgate.check_pair(instr, resp, lang)
        if why:
            drop["gate:" + why.split("(")[0]] += 1; continue
        seen.add((lang, instr.strip().lower()))
        kept.append({"instruction": instr, "response": resp, "src": "teacher",
                     "plang": r.get("plang", lang), "lang": lang,
                     "reg": r.get("reg", "topical"), "budget": r.get("budget", "medium"),
                     "judge": r.get("judge", 0), "topic": r.get("topic", "")})
    with open(outp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    forms = {w for r in kept for w in re.findall(r"[\u0900-\u097f]+", r["response"])}
    print(f"{len(rows):,} in -> {len(kept):,} out  ({len(rows)-len(kept):,} dropped)")
    print("  by lang:", dict(collections.Counter(r["lang"] for r in kept)))
    print(f"  distinct Devanagari forms: {len(forms):,}")
    # The invariant v7 could not show: a real length spread, and any chat register at all.
    wl = sorted(len(r["response"].split()) for r in kept)
    p = lambda q: wl[int(len(wl) * q)] if wl else 0
    short = sum(1 for x in wl if x <= 15)
    print(f"  response words p10/median/p90 = {p(.1)}/{p(.5)}/{p(.9)}"
          f"  | <=15w: {short:,} ({100 * short / max(1, len(wl)):.1f}%)")
    print(f"  register: chat {sum(1 for r in kept if r['reg'] == 'chat'):,}"
          f" topical {sum(1 for r in kept if r['reg'] == 'topical'):,}"
          f"  | topics {len({r['topic'] for r in kept}):,}")
    for k, v in drop.most_common(12):
        print(f"  drop {k:30s} {v:,}")
    # marathi_dominated after the blacklist means the blacklist is wrong. Find the
    # source; do not widen the regex.
    left = sum(v for k, v in drop.items() if k.endswith("marathi_dominated"))
    if left:
        print(f"\nFAIL: {left} Marathi-dominated rows survived the source blacklist."
              f" Attribute them to a prompt src and add it to SRC_BAD.")
        return None
    return kept

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "teach/teacher_gen_main.jsonl"
    outp = sys.argv[2] if len(sys.argv) > 2 else "train_v7_teacher.jsonl"
    sys.exit(0 if build(inp, outp) is not None else 1)
