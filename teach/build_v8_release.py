#!/usr/bin/env python3
"""Combine every refined pool into train_v8.jsonl, then prove it before it ships.

  python3.13 teach/build_v8_release.py           # build + audit + masking selfcheck
  python3.13 teach/build_v8_release.py --no-mask  # skip the GPU-free-but-slow masker pass

WHAT GOES IN, and what is deliberately left out. Each choice is a measurement on this
box, not a preference:

  IN   teach/teacher_gen_v8.jsonl   61,167 rows  13.7M tok  teacher, judge>=4, 50 topics
  IN   teach/oasst_v8.jsonl          5,335 rows  10.7M tok  human multi-turn
  IN   teach/wikihow_v8.jsonl        4,501 rows   7.0M tok  human Hindi how-to + glosses
  OUT  train_v7_teacher.jsonl       57,687 rows  14.1M tok
  OUT  teach/teacher_gen_main.jsonl 100,619 rows  25.8M tok
  OUT  hi_cache/dolly_hi.parquet    10,400 fact rows

v7 and gen_main are the SAME DATA: all 57,687 of v7's (instruction, response) pairs occur
verbatim in gen_main (measured), and gen_main is that file plus 43k rows generated from
the same unsupervised pool. Worse, 26,244 of gen_v8's prompts are the same prompts with a
better, judged answer - so adding v7 does not add breadth, it adds a second and worse
answer to a quarter of the corpus. Its shape is the defect being fixed: response words
p10/med/p90 = 40/47/52, ZERO rows under 15 words, ZERO chat register, and audit_v8.py
fails it on Marathi and dodge rows.

dolly_hi looks like 10,400 free fact rows and is a trap: it carries
backtranslated_instruction/response columns, i.e. English run through a translator, and
the tell is inside single rows - a question asks about एनिहिलेशन and its own answer calls
the same book विनाश. That is the `adaption` pool class that put 1,245 Marathi rows into v6.

THE BALANCE IS IN TOKENS, NOT ROWS. oasst is 8% of rows and 48% of tokens (1,957 tok/row
vs 224 for generated rows); wikihow is 6% of rows and 34%. Cell 22 trains with
`packing = False` and a padding collator, so 15 short rows plus one 4k-token row cost the
batch 16 x 4k positions - a long row is charged for every neighbour it sits with. So each
source gets a share of the TOKEN budget and a per-row ceiling, and where a source is over
share its longest rows go first. Weighting by rows would let two small files own every
gradient step, which is v6's long-verbose collapse waiting to happen again.

Every row is re-gated here - Marathi markers, foreign script, Devanagari->Latin script
mirroring, evasive dodges, cross-pool dedupe - because three files written by three
scripts must not be trusted to agree with each other.
"""
import json, os, re, sys, collections, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import langgate
from audit_v8 import MAR, DODGE, words, devshare, DANGLE          # same gates the audit uses

OUT = os.path.join(ROOT, "train_v8.jsonl")
TOK_DIR = os.path.join(ROOT, "tokenizer-hi")

# Per-row ceiling = the notebook's MAX_SEQ_LENGTH. Rows at 4096 are NOT expensive once
# cell 21 orders batches by length: with packing=False the collator pads each batch to
# its own longest row, so a long row costs its real length and its 15 neighbours cost
# theirs. Measured on this corpus: length-grouped batches need 19.0M positions/epoch vs
# 72.7M for shuffled rows, i.e. 74% less compute, and nothing is dropped. So the cap is
# the model's limit and no lower - cutting to 2048 threw away 2,461 of 5,335 human
# conversations to save padding that the ordering already saved.
MAX_ROW_TOKENS = 4096

# Share of the final TOKEN budget. gen_v8 is the backbone: it is the only source with a
# topic spread, a judged quality signal, romanised Hinglish and short chat rows.
SHARE = {"gen_v8": 0.55, "oasst_v8": 0.30, "wikihow_v8": 0.15}

POOL_FILES = {
    "gen_v8":    os.path.join(HERE, "teacher_gen_v8.jsonl"),
    "oasst_v8":  os.path.join(HERE, "oasst_v8.jsonl"),
    "wikihow_v8": os.path.join(HERE, "wikihow_v8.jsonl"),
}

DEV_Q_MIN = 0.35      # a Devanagari question must not get a Latin answer (v6's bug)


def rows_of(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out


def texts(r):
    """(question, answer, [all turns]) for any row shape."""
    if r.get("messages"):
        m = r["messages"]
        q = next((x["content"] for x in m if x["role"] == "user"), "")
        a = next((x["content"] for x in reversed(m) if x["role"] == "assistant"), "")
        return q, a, [x["content"] for x in m]
    return r["instruction"], r["response"], [r["instruction"], r["response"]]




def gate(r, src):
    q, a, allc = texts(r)
    blob = " ".join(allc)
    if len(set(MAR.findall(blob))) > len(set(langgate.HIN.findall(blob))):
        return "marathi"
    if langgate.FOREIGN.search(blob):
        return "foreign_script"
    if devshare(a) < 0.30 and devshare(q) > 0.35:
        return "script_mirror"          # Devanagari question, romanized answer
    if devshare(a) > 0.35 and devshare(q) < 0.35:
        # The v6 defect in the other direction: a romanized-Hindi question answered in
        # Devanagari teaches the model to switch script on the user, who never asked for
        # it. 18 rows measured, all in the human pools - they are real Hindi, just in the
        # wrong script for the prompt they answer.
        return "script_mirror_reverse"
    if DANGLE.search(a.strip()):
        # Ending on a comma or a postposition is a cut-off generation; training on cut-offs
        # is exactly what teaches the model to stop mid-sentence.
        return "truncated"
    if devshare(blob) < 0.10 and src == "oasst_v8":
        return "no_devanagari"
    if DODGE.search(" ".join(a.split()[:18])):
        return "dodge"
    if len(a.split()) < 4:
        return "too_short"
    return None


def trim_turns(r, tok):
    """Drop the LAST user/assistant pair until the conversation fits the batch budget.

    Deleting an over-long conversation threw away 2,506 of 5,335 oasst rows (47%) - the
    best human-labelled data in the corpus - to solve a padding cost. Trailing turns are
    the cheapest thing to lose: a 6-turn row becomes a 4-turn row, still multi-turn, same
    opener. Never goes below two turns; a row that still cannot fit is dropped, not
    mangled mid-sentence.
    """
    m = list(r["messages"])
    while len(m) > 4:
        m = m[:-2]
        r = dict(r, messages=m, turns=len(m))
        r["tokens"] = len(tok(" ".join(x["content"] for x in m),
                              add_special_tokens=False)["input_ids"])
        if r["tokens"] <= MAX_ROW_TOKENS:
            return r, None
    return r, "still_over_cap"


def build(tok, do_mask=True):
    drop = collections.Counter()
    pools = {}
    for src, path in POOL_FILES.items():
        if not os.path.exists(path):
            sys.exit(f"missing {path} - run its converter first")
        keep = []
        for r in rows_of(path):
            r = dict(r)
            r["src_pool"] = src
            why = gate(r, src)
            if why:
                drop[f"{src}:{why}"] += 1
                continue
            q, a, _ = texts(r)
            if "tokens" not in r:
                r["tokens"] = len(tok(q + "\n" + a, add_special_tokens=False)["input_ids"])
            if r["tokens"] > MAX_ROW_TOKENS:
                if r.get("messages"):
                    r, why = trim_turns(r, tok)
                    if why:
                        drop[f"{src}:{why}"] += 1
                        continue
                else:
                    drop[f"{src}:over_row_cap"] += 1
                    continue
            keep.append(r)
        pools[src] = keep

    # Trim longest-first until the source fits its token share. Shares are of the FINAL
    # corpus, so solve against the fixed backbone: gen_v8 keeps everything it has, and the
    # two long pools are cut relative to it.
    gen = sum(r["tokens"] for r in pools["gen_v8"])
    total_target = gen / SHARE["gen_v8"]
    for src in ("oasst_v8", "wikihow_v8"):
        cap = int(total_target * SHARE[src])
        rows = sorted(pools[src], key=lambda r: -r["tokens"])
        run, out = 0, []
        for r in rows:
            if run + r["tokens"] > cap:
                drop[f"{src}:over_share"] += 1
                continue
            run += r["tokens"]
            out.append(r)
        pools[src] = out

    rows = pools["gen_v8"] + pools["oasst_v8"] + pools["wikihow_v8"]
    seen, final = set(), []
    for r in rows:
        q, a, _ = texts(r)
        k = (q.strip().lower(), a.strip().lower())
        if k in seen:
            drop["duplicate"] += 1
            continue
        seen.add(k)
        final.append(r)

    # deterministic order: single-turn first, conversations after, so a truncated upload
    # still leaves a trainable file, and each group sorted by pool then key.
    final.sort(key=lambda r: (bool(r.get("messages")), r["src_pool"],
                              r.get("key", r.get("instruction", "")[:40])))
    with open(OUT, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    t = sum(r["tokens"] for r in final)
    print(f"wrote {OUT}: {len(final):,} rows, {t/1e6:.1f}M tokens")
    for src in POOL_FILES:
        n = len(pools[src]); st = sum(r["tokens"] for r in pools[src])
        print(f"  {src:12} {n:6,} rows {st/1e6:5.1f}M tok  "
              f"({100*n/len(final):4.1f}% of rows, {100*st/t:4.1f}% of tokens)")
    for k, v in drop.most_common():
        if v:
            print(f"  drop {k:26} {v:,}")
    if do_mask:
        return final, selfcheck(tok, final)
    return final, True


def selfcheck(tok, rows, sample=1200):
    """Render through the notebook's real template and mask with the real unsloth_zoo
    masker. Asserts, per row, that the number of trained runs equals the number of
    assistant turns and each run decodes to that turn. Same contract as
    oasst_to_v8.py --selfcheck, applied to the merged file."""
    sys.path.insert(0, "/tmp/ulz/x")
    os.environ.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    from unsloth_zoo.dataset_utils import train_on_responses_only as zoo
    fn = zoo(None, instruction_part="user\n", response_part="assistant\n",
             tokenizer=tok, return_function=True)

    def runs(lab):
        out, s = [], None
        for i, v in enumerate(lab):
            if v != -100 and s is None:
                s = i
            elif v == -100 and s is not None:
                out.append((s, i)); s = None
        if s is not None:
            out.append((s, len(lab)))
        return out

    bad, over = [], 0
    step = max(1, len(rows) // sample)
    for r in rows[::step]:
        conv = r["messages"] if r.get("messages") else [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]}]
        text = tok.apply_chat_template(conv, tokenize=False,
                                       add_generation_prompt=False).removeprefix(tok.bos_token or "")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_ROW_TOKENS:
            over += 1
            continue
        lab = fn({"input_ids": [ids]})["labels"][0]
        rr = runs(lab)
        want = [m for m in conv if m["role"] == "assistant"]
        if len(rr) != len(want):
            bad.append((r["src_pool"], len(rr), len(want)))
        else:
            for (a, b), m in zip(rr, want):
                if not tok.decode(ids[a:b]).strip().startswith(m["content"].strip()[:24]):
                    bad.append((r["src_pool"], "content", m["content"][:24]))
                    break
    print(f"masking selfcheck over {min(len(rows), sample):,} rows: "
          f"over_cap={over} wrong={len(bad)} {bad[:2]}")
    return not (over or bad)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-mask", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    rows, ok = build(tok, do_mask=not a.no_mask)
    import subprocess
    rc = subprocess.run([sys.executable, os.path.join(HERE, "audit_v8.py"), OUT]).returncode
    print(f"masking={'PASS' if ok else 'FAIL'} audit_rc={rc}")
    sys.exit(0 if (ok and rc == 0) else 1)
