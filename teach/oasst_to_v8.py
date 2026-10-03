#!/usr/bin/env python3
"""hi_cache/oasst1_hi.parquet -> v8 multi-turn rows.

The one HF source on this box that has what v8 lacks: 20,128 Hindi conversations with
median 4 turns (85% multi-turn; v7 has exactly 1 multi-turn row) and human `labels` per
message, so quality filtering does not depend on the teacher grading itself.

Everything else about this corpus is a liability and is handled explicitly here:
  - it is BACKTRANSLATED from English. `lang_mismatch` is a real label on 4% of
    messages, which is the failure mode translationese takes, and we drop those rows.
  - answers run long: median 109 words, p90 253. Most 4-turn conversations blow past
    MAX_SEQ_LENGTH=4096 at the measured 6.08-6.74 tokens/Hindi-word, so length is
    capped on the token estimate, not guessed.
  - 4,236 conversations (UAU) and 1,885 (UAUAU) end on a USER turn. Trained as-is the
    model learns to answer nothing, which is the v7 collapse. Dropped.
  - the same Marathi trap as v6 (4.8% of rows, 100% from `adaption`). Measured 0.0%
    here, but the gate is applied anyway - cheap, and the corpus is regenerated.
  - Devanagari only. It contributes no romanized turns, so the Hinglish register still
    comes from the generator and ds_casual_hinglish.

  python3 teach/oasst_to_v8.py            # writes teach/oasst_v8.jsonl
  python3 teach/oasst_to_v8.py --selfcheck

Self-check renders every conversation through the real LFM2 chat template and runs
unsloth_zoo's real masker over it, asserting that each assistant turn contributes to
the loss and no user turn does. That is the one thing that could silently drop a turn,
so it is verified rather than assumed.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from langgate import MAR, HIN, FOREIGN, dev_ratio   # noqa: E402  (same gate as the generator)

SRC = os.path.join(ROOT, "hi_cache", "oasst1_hi.parquet")
OUT = os.path.join(HERE, "oasst_v8.jsonl")
TOK_DIR = os.path.join(ROOT, "tokenizer-hi")

# Notebook cell 22 runs train_on_responses_only on this pair. Verified against the real
# unsloth_zoo masker + real LFM2 tokenizer: both forms mask identically and correctly.
INSTRUCTION_PART = "user\n"
RESPONSE_PART = "assistant\n"

# Must match the notebook's SFT MAX_SEQ_LENGTH. Length is measured with the real tokenizer
# rather than estimated from words: Hindi costs 6.08-6.74 tokens/word on this vocabulary
# (64,402 entries, 5 single-char Devanagari symbols), and a word-based estimate is off by 2x.
MAX_SEQ_LENGTH = 4096

# Every label field is a float mean of human votes in 0..1, never a bool. An earlier draft
# of this file tested them for truthiness, which counts a single 0.33 dissent as a rejection;
# measured, that flagged 4% of messages as lang_mismatch when only 4 in 6,500 actually were.
# So: thresholds, and only on fields whose high value means "unusable".
REJECT = {"fails_task": 0.5, "not_appropriate": 0.5, "spam": 0.5,
          "lang_mismatch": 0.5, "pii": 0.5, "sexual_content": 0.5, "hate_speech": 0.5}
# Inclusion signal. quality/helpfulness are the human-rated version of what the teacher
# self-judge could not do (calibrated here: dodge 3-4, clean 3-5, full overlap). Rows with
# no quality vote are kept - only ~half the messages are rated at all.
MIN_QUALITY = 0.5
MIN_MESSAGES = 4       # 2 full user/assistant exchanges. 2 messages is a single-turn row,
                       # which is what v7 already has 57,687 of; multi-turn is the reason
                       # this corpus is worth converting at all.
MAX_MESSAGES = 8       # cap on conversation length, keeps rows comparable


def _val(msg, field):
    """Mean vote for `field`, or None if nobody rated it."""
    v = (msg.get("labels") or {}).get(field)
    if isinstance(v, dict):
        v = v.get("value")
    return None if v is None else float(v)


def flagged(msg):
    return any((_val(msg, k) or 0.0) >= t for k, t in REJECT.items())


def clean_conv(msgs):
    """Drop the conversation, not the message. Trimming a message out of a dialogue
    leaves an assistant replying to something it never saw, which is exactly the
    incoherent-exchange failure we are trying to remove.

    Returns (conv, reason): reason is None on success, a named string on rejection. The
    first version of this file returned a bare None and counted one "dropped_structure"
    bucket, which reported 56% loss without saying which rule cost the rows."""
    for m in msgs:
        text = str(m.get("content") or "").strip()
        if not text:
            return None, "empty"
        if m.get("deleted"):
            return None, "deleted"
        if FOREIGN.search(text):
            return None, "foreign_script"
        if flagged(m):
            return None, "human_label_reject"
        # A human-rated-bad assistant turn takes the conversation with it. This is the
        # rated signal the teacher self-judge could not reproduce (calibrated on this box:
        # dodge 3-4, clean 3-5, full overlap). Unrated messages are kept - only about half
        # of all messages carry a quality vote.
        if m.get("role") == "assistant":
            q = _val(m, "quality")
            if q is not None and q < MIN_QUALITY:
                return None, "low_quality_vote"
    if msgs[-1].get("role") != "assistant":
        return None, "ends_on_user"          # would train the model to answer nothing
    if msgs[0].get("role") != "user":
        return None, "starts_on_assistant"
    for a, b in zip(msgs, msgs[1:]):
        if a.get("role") == b.get("role"):
            return None, "not_alternating"
    return [{"role": m["role"], "content": str(m["content"]).strip()} for m in msgs], None


def marathi(text):
    """Same comparison the notebook and the generator use: distinct marker forms."""
    return len({x for x in MAR.findall(text)}) > len({x for x in HIN.findall(text)})


def render(tok, conv):
    """The exact string the notebook's cell 21 will produce, so the length filter is
    measured on the same text the trainer sees."""
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    return text.removeprefix(tok.bos_token or "")


def n_tokens(tok, conv):
    return len(tok(render(tok, conv), add_special_tokens=False)["input_ids"])


def convert():
    import pandas as pd   # python3.13 on this box
    df = pd.read_parquet(SRC)
    stats = {}
    rows, seen = [], set()

    def bump(k, n=1):
        stats[k] = stats.get(k, 0) + n

    for raw in df["messages"]:
        conv, why = clean_conv([dict(m) for m in raw])
        bump("in")
        if conv is None:
            bump("drop_" + why)
            continue
        if not (MIN_MESSAGES <= len(conv) <= MAX_MESSAGES):
            bump("dropped_length")
            continue
        if any(marathi(m["content"]) for m in conv):
            bump("dropped_marathi")
            continue
        if any(dev_ratio(m["content"]) < 0.5 for m in conv):
            bump("dropped_script")
            continue
        key = hashlib.sha1(
            "".join(m["content"] for m in conv).encode("utf-8")).hexdigest()[:16]
        if key in seen:
            bump("dropped_dupe")
            continue
        seen.add(key)
        rows.append({
            "key": key,
            "messages": conv,                 # canonical field the notebook renders
            "instruction": conv[-2]["content"],   # last user turn, for auditing only
            "response": conv[-1]["content"],     # last assistant turn
            "src": "oasst1_hi",
            "reg": "multiturn",
            "plang": "hi",
            "lang": "hindi",
            "turns": len(conv) // 2,
        })
        bump("kept_before_length")

    # Length is measured with the real tokenizer, not estimated from words. The first
    # version of this file used TOKENS_PER_WORD=3.0*0.9 and selfcheck caught 60 of 400
    # rows over the cap: Hindi is 6.08-6.74 tokens/word on this tokenizer (5 single-char
    # Devanagari symbols in a 64,402 vocab), so a word-based estimate is wrong by 2x.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    kept_len, ratios = [], []
    for r in rows:
        conv = r["messages"]
        n = n_tokens(tok, conv)
        words = sum(len(m["content"].split()) for m in conv)
        if words:
            ratios.append(n / words)
        if n > MAX_SEQ_LENGTH:
            bump("dropped_too_long")
            continue
        r["tokens"] = n
        kept_len.append(r)
    rows = kept_len
    if ratios:
        ratios.sort()
        print(f"  measured tokens/word p50={ratios[len(ratios)//2]:.2f} "
              f"p95={ratios[int(len(ratios)*.95)]:.2f} max={ratios[-1]:.2f}")

    rows.sort(key=lambda r: r["key"])           # deterministic output
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {OUT}")
    print(f"  conversations in : {stats.get('in',0):,}")
    for k in sorted(stats):
        if k.startswith("drop_"):
            print(f"  {k:24} {stats[k]:6,} ({100*stats[k]/max(1,stats.get('in',1)):.1f}%)")
    for k in ("dropped_length", "dropped_marathi", "dropped_script",
              "dropped_too_long", "dropped_dupe", "kept_before_length", "kept"):
        if k in stats:
            print(f"  {k:18} {stats[k]:6,} ({100*stats[k]/max(1,stats.get('in',1)):.1f}%)")
    if rows:
        turns = sorted(r["turns"] for r in rows)
        tk = sorted(r["tokens"] for r in rows)
        print(f"  turns/row  p10={turns[len(turns)//10]} median={turns[len(turns)//2]} "
              f"max={turns[-1]}")
        print(f"  real tokens p50={tk[len(tk)//2]} p95={tk[int(len(tk)*.95)]} max={tk[-1]} "
              f"(cap {MAX_SEQ_LENGTH})")
    return rows


def selfcheck(rows):
    """Render through the real chat template, mask with the real unsloth_zoo masker,
    and assert the result is structurally right.

    The first version checked "is this turn's text in the trained span" by string match.
    It was wrong twice over: decoding token ids one at a time splits Devanagari graphemes
    (692 false failures), and rows where the assistant ECHOES the user ("नमस्ते।" ->
    "नमस्ते।") are matched in both places, so a user turn looked unmasked (16 false leaks).
    Masking is a property of the token spans, not the text, so this asserts on spans:
      - trained label runs == number of assistant turns, in order,
      - each run decodes to its assistant turn (plus terminator),
      - nothing masked / everything masked is a hard fail,
      - no row exceeds MAX_SEQ_LENGTH.
    """
    from transformers import AutoTokenizer
    sys.path.insert(0, "/tmp/ulz/x")
    os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from unsloth_zoo.dataset_utils import train_on_responses_only as zoo

    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    fn = zoo(None, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
             tokenizer=tok, return_function=True)

    def runs_of(lab):
        out, start = [], None
        for i, v in enumerate(lab):
            if v != -100 and start is None:
                start = i
            elif v == -100 and start is not None:
                out.append((start, i))
                start = None
        if start is not None:
            out.append((start, len(lab)))
        return out

    over, bad = [], []
    for r in rows:
        conv = r["messages"]
        text = render(tok, conv)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_SEQ_LENGTH:
            over.append((r["key"], len(ids)))
            continue
        lab = fn({"input_ids": [ids]})["labels"][0]
        masked = sum(1 for x in lab if x == -100)
        if masked == 0 or masked == len(lab):
            bad.append((r["key"], f"masked={masked}/{len(lab)}"))
            continue
        want = [m for m in conv if m["role"] == "assistant"]
        runs = runs_of(lab)
        if len(runs) != len(want):
            bad.append((r["key"], f"runs={len(runs)} assistant_turns={len(want)}"))
            continue
        for (a, b), m in zip(runs, want):
            got = tok.decode(ids[a:b]).strip()
            if not got.startswith(m["content"].strip()[:30]):
                bad.append((r["key"], f"run {got[:32]!r} != {m['content'][:32]!r}"))
                break

    print(f"selfcheck over {len(rows):,} rows with the real template + real masker")
    print(f"  rows over {MAX_SEQ_LENGTH} tokens : {len(over)} {over[:3]}")
    print(f"  masking structure wrong: {len(bad)} {bad[:3]}")
    ok = not (over or bad)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="render + mask with the real tokenizer and unsloth_zoo masker")
    ap.add_argument("--limit", type=int, default=0, help="only check the first N rows")
    a = ap.parse_args()
    r = convert()
    if a.selfcheck:
        sys.exit(0 if selfcheck(r[:a.limit] if a.limit else r) else 1)