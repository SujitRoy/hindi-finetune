#!/usr/bin/env python3
"""Print the identity and shape of every input a long run consumes.

Written after a multi-hour CPT run launched on a tokenizer with zero Devanagari
tokens. `extend_tokenizer.py` had existed in the repo for sessions and was never
called from the notebook, so nothing in the log could distinguish "the tokenizer I
wrote a script for" from "the base tokenizer" - both print 64,400 tokens and both
compile.

The rule this enforces: before a run that costs hours, name and measure every
artifact. Not because the values are wrong, but because a wrong value that looks
right is indistinguishable from a right one until it is expensive.

Run standalone:  python3 tests/provenance.py
"""
import hashlib, json, os, re, statistics, sys

HF = "LiquidAI/LFM2.5-1.2B-Instruct"
DEV_LO, DEV_HI = "ऀ", "ॿ"


def rule(t):
    print(f"\n{'─'*74}\n  {t}\n{'─'*74}")


def sha(path, n=1 << 20):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    rule("TOKENIZER")
    try:
        from transformers import AutoTokenizer
        for label, path in (("base", HF),
                            ("notebook TOK_DIR", os.environ.get("TOK_DIR", "/kaggle/working/tok"))):
            try:
                tk = AutoTokenizer.from_pretrained(path)
            except Exception as e:
                print(f"  {label:18s} {path}\n  {'':18s} LOAD FAILED: {type(e).__name__}")
                continue
            v = tk.get_vocab()
            dev = sum(1 for t in v if any(DEV_LO <= c <= DEV_HI for c in t))
            mark = "OK" if dev else "  <-- ZERO Devanagari, Hindi is byte-fallback"
            print(f"  {label:18s} {path}")
            print(f"  {'':18s} vocab {len(v):,}  Devanagari {dev:,}   {mark}")
    except ImportError:
        print("  transformers not installed here; run inside the notebook to see this")

    # Parse the run notebook FIRST, so DATA reports the file the run consumes. It used to
    # list v5/v6 by hardcoded name - a provenance report that never mentioned the corpus
    # being trained on, and silently printed v6's numbers for a v8 run.
    NB = sys.argv[1] if len(sys.argv) > 1 else os.path.join("notebooks", "05-hindi-sft-v8.ipynb")
    nb_path = NB if os.path.isabs(NB) else os.path.join(root, NB)
    nb = json.load(open(nb_path))
    src = "\n".join("\n".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

    def g(name):
        m = re.search(rf'^{name}\s*=\s*(.+?)(?:\s+#.*)?$', src, re.M)
        return m.group(1).strip() if m else "-"

    def strip_q(v):
        return v.strip("\"'") if v != "-" else v

    rule("NOTEBOOK")
    print(f"  {os.path.relpath(nb_path, root)}")

    rule("DATA")
    names = [strip_q(g("DATA_FILE")), strip_q(g("CPT_FILE")), "train_v6_teacher.jsonl"]
    for name in [n for n in names if n != "-"]:
        p = os.path.join(root, name)
        if not os.path.exists(p):
            print(f"  {name:26s} not on this box (fetched from the Hub on Kaggle)")
            continue
        rows, lens, form = 0, [], set()
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows += 1
                form.add(tuple(sorted(r.keys())))
                lens.append(len((r.get("response") or r.get("text") or "").split()))
        d = "sha256:" + sha(p)
        print(f"  {name:26s} sha {d}")
        print(f"  {'':26s} rows {rows:,}  fields {sorted(list(form)[0])}")
        if lens:
            lens.sort()
            print(f"  {'':26s} words  median {lens[len(lens)//2]}  p90 {lens[int(len(lens)*.9)]}  max {lens[-1]}")

    rule("BASE + CONFIG")
    # Constants come from the notebook parsed above, not from 04-teacher-distill.
    try:
        al, r = 0, 1
        for k in ("BRANCH", "BASE_REPO", "BASE_BRANCH", "RUN_CPT", "CPT_STEPS", "CPT_SEQ",
                  "CPT_R", "CPT_ALPHA", "CPT_LR", "CPT_FILE", "MAX_SEQ_LENGTH",
                  "LEARNING_RATE", "MAX_STEPS", "DTYPE", "DATA_FILE"):
            print(f"  {k:16s} {g(k)[:70]}")
        m = re.search(r"model, r = (\d+), lora_alpha = (\d+), lora_dropout = ([\d.]+)", src)
        if m:
            r, al, dr = int(m.group(1)), int(m.group(2)), float(m.group(3))
            print(f"  {'SFT adapter':16s} r={r} alpha={al} dropout={dr}  scale alpha/r = {al/r:.2f}")
        m = re.search(r"random_state = 3407, use_rslora = (\w+)", src)
        if m:
            print(f"  {'SFT use_rslora':16s} {m.group(1)}"
                  + (f"  -> scale alpha/sqrt(r) = {al/(r**0.5):.2f}" if m.group(1) == "True" else ""))
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

    rule("VERDICT")
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(HF)
        dev = sum(1 for t in tk.get_vocab() if any(DEV_LO <= c <= DEV_HI for c in t))
        if dev == 0:
            print("  The notebook does not extend the tokenizer, so Hindi rides byte-fallback.")
            print("  embed_tokens must stay in CPT_TARGETS or those bytes never learn a meaning.")
            print("  extend_tokenizer.py exists and is not called. That is a known, accepted gap.")
        else:
            print(f"  {dev:,} Devanagari tokens present. Extended tokenizer is wired in.")
    except Exception:
        pass
    print()


if __name__ == "__main__":
    sys.exit(main())
