#!/usr/bin/env python3
"""The Hindi/Hinglish language gate. One rule set, used by the teacher generator
BEFORE it spends API calls and by the release builder BEFORE a corpus ships.

Why this exists: the Devanagari rows are not all Hindi. Marathi is written in the
same script with different grammar (आहे/आणी/नाही/झाले for है/और/नहीं/हुआ, and a
verb that agrees with the object instead of the subject). A corpus that is a few
percent Marathi teaches the student to answer Hindi questions with Marathi
inflection, which is exactly the "fluent but wrong grammar" the model shows.
`build_cpt_clean.py` already gated the CPT corpus for this. The SFT path never
had the same gate, and `tests/preflight.py` has no language check in it, so it
shipped.

Markers are WORD-ANCHORED on both sides. An unanchored `मला` matches inside
`दिल्ली`, `छे` inside `अच्छे`, `काय` inside `कार्य` - all Hindi. That false
positive rate was measured and documented in build_cpt_clean.py; do not remove
the anchors.
"""
import re

DEV_LO, DEV_HI = "\u0900", "\u097f"

# Confirmed Marathi-only, never used in Hindi prose. Anchored so कायम, कार्य,
# अच्छे and दिल्ली cannot match. Calibrated against ds_hindi_devanagari.jsonl
# (6,000 known-Hindi texts): every form below matched ZERO of them, except that
# करते and सकते were dropped from an earlier draft of this list - they are
# ordinary Hindi inflections (करते हैं, बता सकते हैं) and hit 479 and 660 of those
# texts. Marathi uses करतो / शकतो, which stay. Re-measure before adding a form.
MAR = re.compile(
    r"(?<![\u0900-\u097f])(आहे|आणि|तुम्ही|तुम्हाला|नाही|पण|मला|माझे|माझा|माझी|"
    r"आमचा|आमची|झाले|झाला|झाली|काय|कोणता|कोणती|म्हणजे|मुळे|याचा|याची|दिला|दिली|"
    r"घेतो|करतो|सकतो|असतो|असेल|आहोत)(?![\u0900-\u097f])")
# Hindi function words. A Marathi document has almost none; a Hindi one has many.
HIN = re.compile(
    r"(?<![\u0900-\u097f])(है|हैं|हूँ|और|क्या|में|से|के|लिए|लेकिन|नहीं|हुआ|होता|करना|"
    r"किया|गया|मिला|दिया|आपका|आपकी|अपना|अपनी|सकता|सकती|रहा|रही|क्यों|कौन)(?![\u0900-\u097f])")
# Romanised Marathi, for the Hinglish bucket. `nahi` is deliberately excluded:
# it is the normal romanisation of Hindi नहीं as well.
MAR_ROM = re.compile(
    r"\b(aahe|aani|tumhi|tumhala|mazha|mazi|maje|majhe|zala|zale|zali|mhanun|"
    r"mhanje|mulane|yacha|yachi|dila|dili|ghetlo|kartoy|astey|amha|asmhi)\b", re.I)
# Anything that is not Devanagari, Latin, digits or punctuation is a foreign script
# that leaked in through a machine-translated source (measured: CJK inside a
# "Marathi" prompt, Bengali/Gujarati slivers elsewhere).
FOREIGN = re.compile(r"[\u0980-\u09ff\u0a80-\u0aff\u0b80-\u0d7f\u3040-\u30ff"
                     r"\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff]")

def dev_ratio(t):
    return sum(1 for c in t if DEV_LO <= c <= DEV_HI) / max(1, len(t))

def marathi_score(text):
    """(marathi markers, hindi markers). Compare them; presence alone is not proof,
    a Hindi document can carry one coincidental match."""
    return len(MAR.findall(text)), len(HIN.findall(text))

def _stem_hits(rx, text):
    """Count matches per distinct surface form, not per occurrence: one word
    repeated across a paragraph is one signal, not ten."""
    return len({m for m in rx.findall(text)})

def reject(text, lang):
    """None = keep. Otherwise the reason it never reaches training."""
    if not text or not text.strip():
        return "empty"
    if FOREIGN.search(text):
        return "foreign_script"
    m, h = _stem_hits(MAR, text), _stem_hits(HIN, text)
    if lang == "hindi":
        if m > h:
            return f"marathi_dominated({m}>{h})"
        if m >= 3:
            return f"marathi_markers({m})"
    if lang == "hinglish" and len(MAR_ROM.findall(text)) >= 2:
        return "marathi_romanised"
    return None

def check_pair(instr, resp, lang):
    """Gate both sides: a Marathi QUESTION is what recruits a Marathi answer, so a
    row that only looks clean on the response side is still poison."""
    return reject(resp, lang) or reject(instr, lang)

if __name__ == "__main__":
    import json, sys, collections
    src = sys.argv[1] if len(sys.argv) > 1 else "train_v6_teacher.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    drop = collections.Counter(); keep = collections.Counter()
    for r in rows:
        lang = r.get("lang", "hindi")
        why = check_pair(r.get("instruction", ""), r.get("response", ""), lang)
        if why:
            drop[why.split("(")[0]] += 1
        else:
            keep[lang] += 1
    tot = len(rows); bad = sum(drop.values())
    print(f"{src}: {tot:,} rows -> keep {tot - bad:,}  drop {bad:,} ({100 * bad / max(1, tot):.1f}%)")
    print("  kept by lang:", dict(keep))
    for why, n in drop.most_common(10):
        print(f"  drop {why:34s} {n:,}")
