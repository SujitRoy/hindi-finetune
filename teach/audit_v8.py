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

# The v8 shape, taken from generate.py rather than invented here. BUDGETS["short"] is
# (8, 22, ...) - a compliant short row may be 22 words, and the measured p50 of the
# 24,396 budget=short rows is 19. Demanding "20% of rows <= 15 words" therefore fails a
# corpus that was built correctly: <=15 gives 9.6%, <=22 gives 32.3%. The bar is the band.
TARGETS = {
    "short_share":   0.20,   # share of rows at or under SHORT_MAX words
    "chat_share":    0.10,   # reg == "chat"
    "max_topic_p":   0.35,   # no single bucket may dominate (see TOPIC_BUCKETS)
}
SHORT_MAX = 22               # upper bound of BUDGETS["short"] in teach/generate.py

DEV   = re.compile(r"[\u0900-\u097f]")
WORD  = re.compile(r"[A-Za-z\u0900-\u097f']+")
DODGE = re.compile(r"(ka naam[^.]{0,40}batayein|pin ?code[^.]{0,30}batayein|"
                   r"locality[^.]{0,30}batayein|context[^.]{0,40}(batayein|share karein)|"
                   r"apni baat[^.]{0,40}share karein)", re.I)
MAR   = re.compile(r"(?<![\u0900-\u097f])(आहे|आणि|तुम्ही|तुम्हाला|झाले|म्हणजे|मुळे|सकतो|असतो)"
                   r"(?![\u0900-\u097f])")

HIN   = re.compile(r"(?<![\u0900-\u097f])(है|हैं|हूँ|और|क्या|में|से|के|लिए|नहीं|हुआ|होता|"
                   r"करना|किया|गया|मिला|दिया|सकता|सकती|रहा|रही)(?![\u0900-\u097f])")

def words(s): return len(WORD.findall(s))

# The `topic` column cannot answer "is the corpus all about one thing": build_prompts_v8
# writes `r.get("category") or r.get("src")`, and the 105k topical pool has no category, so
# 62% of rows read "teacher_pool_v2" - that is provenance, not subject. So bucket by
# keyword instead. Deliberately coarse and local: a mis-bucketed row only moves a bar
# chart, and the point is to catch a 60%-of-corpus subject, not to classify 66k texts.
TOPIC_BUCKETS = {
    # BOTH scripts in every bucket (46% of this corpus answers in Devanagari, so an
    # English-only pattern under-counts exactly the rows it should count) and \b on BOTH
    # sides of every short Latin word. The first version of this list wrote "ai\b" and
    # matched "hai" - Hindi "is" - inside a third of the corpus, which reported a
    # nonexistent 38% tech concentration.
    "tech":      r"\b(computer|laptop|phone|mobile|internet|wifi|software|python|coding|"
                 r"programming|email|gmail|whatsapp|instagram|youtube|server|database|"
                 r"machine learning|gaming|arduino|router|html|javascript|android|iphone|"
                 r"app|apps|code|ai|it|url|pdf|usb|gpu|cpu)\b"
                 r"|(कंप्यूटर|मोबाइल|फ़ोन|इंटरनेट|सॉफ्टवेयर|प्रोग्राम|कोड|एप|ईमेल|व्हाट्सएप|"
                 r"यूट्यूब|इंस्टाग्राम|लैपटॉप|डेटाबेस|आर्टिफिशल|मशीन लर्निंग|टेक्नोलॉजी|वेबसाइट)",
    "money":     r"\b(money|paisa|salary|loan|emi|bank|invest|shares|mutual fund|tax|gst|"
                 r"budget|expenses|savings|credit card|debit|inflation|interest|fd|rd|upi|"
                 r"cash|income|price|prices|cost|pension)\b"
                 r"|(पैसा|वेतन|लोन|बैंक|निवेश|शेयर|टैक्स|बजट|बचत|महंगाई|सैलरी|कीमत|दाम|मुनाफा)",
    "health":    r"\b(health|disease|doctor|medicine|pain|fever|exercise|workout|gym|diet|"
                 r"weight|sleep|stress|anxiety|mental|therapy|hospital|symptom|vitamin|"
                 r"protein|calories|dietary|nutrition|yoga)\b"
                 r"|(स्वास्थ्य|बीमारी|डॉक्टर|दवाई|दवा|बुखार|व्यायाम|कसरत|आहार|नींद|तनाव|"
                 r"मानसिक|थेरेपी|अस्पताल|लक्षण|विटामिन|प्रोटीन|योग|मोटापा|वजन)",
    "career":    r"\b(job|naukri|interview|resume|career|promotion|boss|colleague|office|"
                 r"internship|exam|exams|study|college|school|admission|course|student|"
                 r"teacher|education|scholarship|syllabus)\b"
                 r"|(नौकरी|इंटरव्यू|करियर|पदोन्नति|कार्यालय|परीक्षा|पढ़ाई|कॉलेज|स्कूल|"
                 r"प्रवेश|छात्र|अध्ययन|शिक्षक|पैठ्यक्रम)",
    "relationships": r"\b(friend|dost|girlfriend|boyfriend|family|parents|mother|father|"
                 r"brother|sister|marriage|shaadi|wife|husband|love|breakup|argue|argument|"
                 r"respect|trust|apologize|sorry|compliment|flirt|relationship|children|"
                 r"cousin|relative| neighbours|neighbours)\b"
                 r"|(दोस्त|परिवार|माता|पिता|भाई|बहन|शादी|पत्नी|पति|प्यार|विश्वास|सम्मान|"
                 r"क्षमा|तर्क|रिश्ता|रिश्ते|बच्चे|संबंध)",
    "travel":    r"\b(travel|train|flight|bus|hotel|ticket|passport|visa|trip|tourism|"
                 r"station|airport|baggage|route|directions|tourist|holiday|vacation)\b"
                 r"|(यात्रा|ट्रेन|फ्लाइट|बस|होटल|टिकट|पासपोर्ट|वीजा|सैर|पर्यटन|स्टेशन|"
                 r"हवाई अड्डा|रास्ता|छुट्टी)",
    "law":       r"\b(law|legal|court|police|fir|rights|contract|agreement|complaint|"
                 r"cyber|fraud|scam|copyright|privacy|ruling|penalty|act\b|section\b|"
                 r"lawyer|case)\b"
                 r"|(कानून|कानूनी|अदालत|पुलिस|शिकायत|संविदा|साइबर|धोखा|गोपनीयता|निर्णय|"
                 r"जुर्माना|वकील|अधिकार)",
    "food":      r"\b(recipe|recipes|khana|food|cooking|dish|dishes|snack|tea|coffee|"
                 r"breakfast|dinner|lunch|vegetable|vegetables|fruit|fruits|masala|spices|"
                 r"drink|drinks)\b"
                 r"|(खाना|भोजन|रेसिपी|पकाना|व्यंजन|नाश्ता|रात का खाना|सब्जी|फल|मसाला|पेय)",
    "animals":   r"\b(dog|cats|dog\b|pet|birds|cows|animals|fish|aquarium|horses)\b"
                 r"|(कुत्ता|बिल्ली|पशु|पक्षी|मछली|जानवर|पालतू)",
}
_B = {k: re.compile(v, re.I) for k, v in TOPIC_BUCKETS.items() if v}


def bucket(r):
    """Keyword subject over question + answer head. 'other' = no hit.

    Returns the FIRST hit, so a row about UPI PINs at a bank counts once, under whichever
    bucket its regex reaches first. That is a coarse read of the corpus and it is enough:
    the check exists to catch one subject owning the file, and the multi-hit rows are
    printed by --buckets so an ambiguous bucket is visible rather than hidden.
    """
    q, a = (r["instruction"], r["response"]) if not r.get("messages") else (
        next((m["content"] for m in r["messages"] if m["role"] == "user"), ""), "")
    text = (q + " " + a)[:400]
    hits = [k for k, rx in _B.items() if rx.search(text)]
    return hits[0] if hits else "other"


def is_marathi(r):
    blob = " ".join(m["content"] for m in r["messages"]) if r.get("messages") else r["response"]
    return len(set(MAR.findall(blob))) > len(set(HIN.findall(blob)))
# A response that STOPS MID-STRUCTURE is the signal of a truncated generation. Do NOT
# require a punctuation terminator: 416 rows fail that and only 3 are actually cut off.
# The other 413 are markdown tables (`| row |`), code blocks (`}`), bare URLs, `SHOW
# DATABASES;` and letter sign-offs - complete answers with no full stop. A terminator rule
# would delete 268 of the corpus's best-formatted wiki tables to satisfy a heuristic written
# for sentence prose. This matches only a genuinely dangling tail.
DANGLE = re.compile(
    r"(?:[,;:]"                                          # trailing comma / semicolon / colon
    r"|\b(?:और|के|का|की|कि|जो|तो|एवं|या|से|में|हुआ|लेकिन|उदाहरण"
    r"|for|and|the|of|to|is|in|with|that|which)\b)\s*$",   # or a dangling connective
    re.I)


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
    short = sum(1 for x in rl if x <= SHORT_MAX)
    chat  = sum(1 for r in rows if r.get("reg") == "chat")
    dodges = sum(1 for r in rows if DODGE.search(" ".join(r["response"].split()[:18])))
    # Dominance, not presence. This used to flag any single marker, which false-fails
    # ordinary Hindi: झाले is the third section of a raga (आलाप-जोर-झाले) and one row in
    # 66k got deleted for it. Every other gate in the repo (langgate, notebook cell 9,
    # build_release) compares distinct Marathi against distinct Hindi markers, so the
    # audit now agrees with the filter it exists to check.
    marathi = sum(1 for r in rows if is_marathi(r))
    # script mirroring: Devanagari question must not get a romanized answer
    mirror = sum(1 for r in rows
                 if devshare(r.get("instruction", "")) > 0.35 and devshare(r["response"]) < 0.35)
    topics = collections.Counter(r.get("topic", "?") for r in rows)
    buckets = collections.Counter(bucket(r) for r in rows)
    judges = collections.Counter(r.get("judge", 0) for r in rows)
    langs  = collections.Counter(r.get("lang", "?") for r in rows)

    print(f"file                {path}")
    print(f"rows                {len(rows):,}")
    print(f"langs               {dict(langs)}")
    print(f"judge scores        {dict(sorted(judges.items()))}")
    print()
    print(f"response words  p10/med/p90   {p(rl,.1)}/{p(rl,.5)}/{p(rl,.9)}   (v7: 40/47/52)")
    print(f"  <= {SHORT_MAX} words  {short:,} ({100*short/len(rows):.1f}%)   target >= "
          f"{100*TARGETS['short_share']:.0f}%   {'OK' if short/len(rows) >= TARGETS['short_share'] else 'LOW'}")
    print(f"  chat register {chat:,} ({100*chat/len(rows):.1f}%)   target >= "
          f"{100*TARGETS['chat_share']:.0f}%   {'OK' if chat/len(rows) >= TARGETS['chat_share'] else 'LOW'}")
    print(f"instruction words p10/med/p90 {p(il,.1)}/{p(il,.5)}/{p(il,.9)}   (v7: 10/13/18)")
    print(f"  <=6 words     {sum(1 for x in il if x <= 6):,} "
          f"({100*sum(1 for x in il if x<=6)/len(il):.1f}%)   v7 was 0.04%")
    print()
    print(f"topics(field)       {len(topics)} distinct - NOTE: for topical prompts this is"
          f" the SOURCE name, not a subject (build_prompts_v8 falls back to r['src'])")
    # 'other' is the residual, so it is excluded from the cap: the check exists to catch a
    # single SUBJECT owning the corpus, not to catch the bucket that means "no keyword hit".
    named = {k: v for k, v in buckets.items() if k != "other"}
    top_b, top_n = max(named.items(), key=lambda kv: kv[1])
    _b = sum(named.values())
    print(f"keyword buckets     {len(named)} subjects, {_b:,} rows bucketed "
          f"({100*_b/len(rows):.0f}%), largest '{top_b}' {top_n/len(rows):.1%} "
          f"(cap {TARGETS['max_topic_p']:.0%})")
    for t, n in sorted(named.items(), key=lambda kv: -kv[1])[:9]:
        print(f"    {n:6,}  {t[:40]}")
    print()
    bad = [("marathi rows", marathi), ("dodge rows", dodges),
           ("script-mirror violations", mirror)]
    for name, n in bad:
        print(f"  {'FAIL' if n else 'ok':4}  {name}: {n:,}")
    fails = [n for _, n in bad if n] or \
            [x for x, ok in [("short_share", short/len(rows) >= TARGETS["short_share"]),
                             ("chat_share", chat/len(rows) >= TARGETS["chat_share"]),
                             ("topic_cap", top_n/len(rows) <= TARGETS["max_topic_p"]),
                             ("bucketed_coverage", _b/len(rows) >= 0.25)] if not ok]
    print("\n" + ("AUDIT PASS" if not fails else f"AUDIT FAIL: {fails}"))
    return 0 if not fails else 1

def topics_iter(rows):
    for r in rows: yield r

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "teacher_gen_v8.jsonl")))
