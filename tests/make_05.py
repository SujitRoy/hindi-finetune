#!/usr/bin/env python3
"""Derive notebooks/05-hindi-sft-v7.ipynb from 04-teacher-distill.ipynb.

Generated, not hand-copied: 04 carries every guard this project paid for in
wasted runs (the deleted training cell, the anchor written and never read,
train_on_responses_only documented but not called, the GPU guard, chdir before
the GGUF intermediate). Re-typing those is how they go missing.

  python3 tests/make_05.py            # writes + validates, non-zero on any FAIL

Changes, all for the Hindi/Hinglish-only scope and the v7 corpus:
  1. cell 1   BRANCH / DATA_FILE -> s3-hinglish-v7 / train_v7_teacher.jsonl
  2. cell 9   language ASSERTION. The source blacklist runs locally in
              teach/build_release.py where the pool files live; on Kaggle the
              corpus only has to prove the blacklist worked. No pool files, no
              network, no new failure mode for a 6h session.
  3. cell 21  English anchor DROPPED. It exists to keep English alive; out of
              scope now, and it hard-raises without prompts_en.json attached.
  4. cell 28  exit instructions carry the vendor decode settings.

Cell source is a LIST OF LINES WITH NO TRAILING NEWLINES except the last. Join
with "\\n" and strip !/% magics before compiling - see commit 5415ea2, which
fixed exactly this in a previous patch.
"""
import json, re, sys

SRC_NB = "notebooks/04-teacher-distill.ipynb"
DST    = "notebooks/05-hindi-sft-v7.ipynb"
CODE_STARTS = ("!", "%")

def cell_src(cell):
    """The repo's own convention, same as tests/preflight.py."""
    return "\n".join(l for l in cell["source"] if not l.strip().startswith(CODE_STARTS))

def split_lines(text):
    """Back to ipynb line-list form, keeping the newline on every line but the last."""
    lines = text.splitlines(keepends=True)
    if lines: lines[-1] = lines[-1].rstrip("\n")
    return lines

nb = json.load(open(SRC_NB, encoding="utf-8"))
cells = nb["cells"]

def src(i): return "".join(l if l.endswith("\n") else l + "\n" for l in cells[i]["source"])
def setsrc(i, text):
    cells[i]["source"] = split_lines(text)
    if cells[i]["cell_type"] == "code":
        cells[i]["outputs"] = []; cells[i]["execution_count"] = None

# ---------------------------------------------------------------- 1. run config
c = src(1)
c = c.replace('BRANCH = "s2c-teacher-10k"', 'BRANCH = "s3-hinglish-v7"')
# FINDINGS 5.1 is this exact trap: "matching their flag without matching their
# budget under-trains". v6 ran 1500 steps on 19,775 rows = 1.21 epochs. v7 has
# 57,687 rows, so 1500 would be 0.42 epochs - less exposure than the run being
# replaced. One epoch is 57,687/16 = 3,606 steps. Run 3,600 and let the eval
# curve decide the real stop: cell 23 prints "next run should stop near step N".
c = re.sub(r"MAX_STEPS\s+= 1500   # ~1\.0 epoch on ~11\.9k rows",
           "MAX_STEPS     = 3600   # ~1.0 epoch of v7: 57,687 rows / 16 per step.\n"
           "#                  # v6 ran 1500 on 19,775 = 1.21 epochs; 1500 here = 0.42.\n"
           "#                  # Stop at the eval-loss minimum the run prints.",
           c, count=1)
assert "MAX_STEPS     = 3600" in c, "MAX_STEPS not rewritten - 04 changed shape, re-slice it"

c = c.replace('DATA_FILE = "train_v6_teacher.jsonl"',
              'DATA_FILE = "train_v7_teacher.jsonl"')
for need in ('BRANCH = "s3-hinglish-v7"', 'train_v7_teacher.jsonl',
             'BASE_BRANCH = "cpt-hindi"', 'No GPU visible'):
    assert need in c, f"cell 1 lost {need!r}"
setsrc(1, c)

# ---------------------------------------------------------------- 2. language assertion
GATE = r'''
# ==========================================================================
# LANGUAGE ASSERTION. Marathi is written in Devanagari, so every existing
# script check passes a poisoned corpus: "hindi rows are >=55% Devanagari" is
# true of Marathi, and 4.8% of v6's Hindi rows were exactly that.
#
# The FIX is a source blacklist and it runs LOCALLY in teach/build_release.py,
# where the prompt pools live (1,245/1,245 Marathi rows = 100% attributable to
# the `adaption` machine-translated pool). What this cell does is prove the
# blacklist worked before spending 6 GPU-hours. No pool files, no network.
#
# The marker list is calibrated, not guessed. Measured against
# ds_hindi_devanagari.jsonl (6,000 known-Hindi texts): करते and सकते hit 479
# and 660 of them because they are ordinary Hindi inflections (करते हैं, बता
# सकते हैं); Marathi uses करतो / शकतो. Unanchored forms match inside Hindi
# words (मला in दिल्ली, काय in कार्य, छे in अच्छे), so both sides are anchored.
# ==========================================================================
_MAR = re.compile(r"(?<![\u0900-\u097f])(आहे|आणि|तुम्ही|तुम्हाला|पण|मला|माझे|माझा|माझी|"
                  r"झाले|झाला|काय|कोणता|कोणती|म्हणजे|मुळे|याच|दिला|दिली|सकतो|असतो|आहोत)"
                  r"(?![\u0900-\u097f])")
_HIN = re.compile(r"(?<![\u0900-\u097f])(है|हैं|हूँ|और|क्या|में|से|के|लिए|नहीं|हुआ|होता|"
                  r"करना|किया|गया|मिला|दिया|सकता|सकती|रहा|रही)(?![\u0900-\u097f])")
_FOREIGN = re.compile("[\u0980-\u09ff\u0a80-\u0aff\u0b80-\u0dff\u3040-\u30ff"
                      "\u4e00-\u9fff\uac00-\ud7af\u0600-\u06ff]")

_mar = [r for r in rows
        if len(set(_MAR.findall(r["response"]))) > len(set(_HIN.findall(r["response"])))]
_foreign = [r for r in rows
            if _FOREIGN.search(r["instruction"]) or _FOREIGN.search(r["response"])]
_nonhh = [r for r in rows if r.get("lang") not in ("hindi", "hinglish")]

print(f"language assertion on {len(rows):,} rows:")
print(f"  marathi-dominated : {len(_mar):,}")
print(f"  foreign script    : {len(_foreign):,}")
print(f"  not hindi/hinglish: {len(_nonhh):,}")
for r in (_mar + _foreign + _nonhh)[:3]:
    print("    e.g.", r.get("lang"), "|", r["instruction"][:46], "->", r["response"][:46])

# Hard stops. Each one is a 6-hour run wasted and a model that still sounds wrong.
assert not _mar, (
    f"{len(_mar)} Marathi-dominated rows in {DATA_FILE}. This file is supposed to be"
    " the OUTPUT of teach/build_release.py, which blacklists the source. Rebuild the"
    " release locally; do NOT widen this regex to make the run go through.")
assert not _nonhh, (
    f"{len(_nonhh)} rows are not hindi/hinglish (scope is those two only)."
    " Rebuild with teach/build_release.py.")
# foreign script is scraped garbage in small volume, filtered here rather than
# failed on: 97 rows measured, and it cannot be fixed by re-pulling a source.
if _foreign:
    rows = [r for r in rows if r not in _foreign]
    print(f"  dropped {len(_foreign)} foreign-script rows; continuing")
assert len(rows) > 5000, f"only {len(rows)} rows survive; expected ~56k for v7"
_hi = sum(1 for r in rows if r["lang"] == "hindi")
_forms = len({w for r in rows for w in re.findall("[\\u0900-\\u097f]+", r["response"])})
print(f"FINAL: {len(rows):,} rows | hindi {_hi:,} hinglish {len(rows)-_hi:,} | "
      f"distinct Devanagari forms {_forms:,}")
'''
c9 = src(9)
# Keep the whole DATA_PATH resolution block (CANDIDATES -> local, Kaggle input,
# private HF dataset repo, GitHub last). Slicing from `rows = ` instead drops it
# and tests/audit_standalone.py correctly fails the cell on a missing name.
i_repo = c9.index('DATA_REPO = "')
i_eng  = c9.index('eng = lang.get("english", 0)')
setsrc(9, "import os, glob, json, re\n" + c9[i_repo:i_eng] + GATE)

# ---------------------------------------------------------------- 3. drop the English anchor
# Two places: cell 21 loads the anchor file, cell 19 GENERATES it with the base model.
# v6 was rescued from English collapse by that anchor. v7 is Hindi/Hinglish-only by
# product decision, so the anchor adds English rows back AND hard-raises when
# prompts_en.json is not attached. Delete the generation block, keep its imports and
# everything from the LoRA attach onwards.
c21 = src(21)
setsrc(21, c21[:c21.index("# The English anchor is self-distilled")]
           + c21[c21.index("ds = Dataset.from_list(rows)"):])

c19 = src(19)
i0 = c19.index("# English anchor by SELF-DISTILLATION")
i1 = c19.index("# ---- attach the adapter only now")
_keep_imports = "import os, re, json, glob, time, urllib.request, torch\n"
assert _keep_imports in c19[i0:i1], "anchor imports moved, re-check the slice"
# imports live in 04-teacher-distill cell 19 already, so they are not lost by the cut
setsrc(19, c19[:i0] + _keep_imports + "\n" + c19[i1:])
assert "english_anchor" not in src(19) and "get_peft_model" in src(19)

# ---------------------------------------------------------------- 3b. title
setsrc(0, "# Experiment 5 - Hindi + Hinglish SFT on the v7 clean corpus\n\n"
    "Chained: CPT on Hindi (`cpt-hindi`) -> SFT on `train_v7_teacher.jsonl`. Single\n"
    "Kaggle session, T4 x2, resumable via `wip/s3-hinglish-v7` every 300 s.\n\n"
    "**What changed from 04.** v6 carried 4.8% Marathi in the Hindi rows - all of it\n"
    "from one machine-translated prompt source. v7 blacklists that source (100%\n"
    "attribution, 1,245/1,245) and this notebook ASSERTS the result before training.\n"
    "English output and the English self-distillation anchor are dropped: the product\n"
    "is Hindi + Hinglish only. Script is not the problem - Devanagari was 5/5 after\n"
    "CPT; grammar was, and grammar lives in these labels.\n\n"
    "**Do not run cell 9 past its assertion.** Rebuild the corpus locally with\n"
    "`teach/build_release.py` instead of editing the regex - see FINDINGS.md 13.")

# ---------------------------------------------------------------- 4. stale v6 comment
c8 = src(8).replace("train_v6_teacher.jsonl; cell 9 fetches it",
                   "train_v7_teacher.jsonl; cell 9 fetches it")
setsrc(8, c8)

# ---------------------------------------------------------------- 5. exit instructions
setsrc(28, src(28).rstrip() + (
    "\n\nVendor decode settings (model card + docs.liquid.ai prompting guide), put these on\n"
    "the llama-server line, not in chat.sh:\n\n```\n--temp 0.1 --top-k 50 --min-p 0.15 --repeat-penalty 1.05\n```\n\n"
    "chat.sh defaults to temp=1.0 and a jailbreak system prompt. Measured on this box:\n"
    "greedy/t=0.5 collapses replies to ~5 words, t=1.0 rambles, and the vendor card sits\n"
    "between them at 0.1. Sampling is polish, not the grammar fix; the grammar fix is v7.\n"))

def _order_ok(book):
    """Position of the `import unsloth` STATEMENT vs the first transformers import
    STATEMENT, line-anchored. A substring search hits the prose comment 'Every
    later `from transformers import ...` is then safe', which lives in the
    unsloth cell itself and makes any naive check fail."""
    u = t = None
    for i, c in enumerate(book["cells"]):
        if c["cell_type"] != "code": continue
        for ln, line in enumerate(cell_src(c).splitlines()):
            st = line.strip()
            if u is None and st == "import unsloth": u = (i, ln)
            if t is None and (st.startswith("from transformers")
                              or st.startswith("import transformers")): t = (i, ln)
        if u and t: break
    return u is not None and t is not None and u < t


def _regex_probe(book):
    """Run the notebook's OWN regexes over known-Hindi and known-Marathi text.
    The false positives that made करते/सकते unusable are the reason this exists,
    so a green build that silently reintroduces them must not pass."""
    import re
    cell = next(c for c in book["cells"] if c["cell_type"] == "code"
                and "_MAR = re.compile" in cell_src(c))
    ns = {"re": re}
    for line in cell_src(cell).splitlines():
        if line.startswith(("_MAR", "_HIN", "_FOREIGN")) or line.lstrip().startswith('r"'):
            pass
    m = re.search(r"_MAR = re\.compile\((.*?)\)\n", cell_src(cell), re.S)
    h = re.search(r"_HIN = re\.compile\((.*?)\)\n", cell_src(cell), re.S)
    exec(f"_MAR = re.compile({m.group(1)})", ns)
    exec(f"_HIN = re.compile({h.group(1)})", ns)
    def mar(t): return len(set(ns["_MAR"].findall(t))) > len(set(ns["_HIN"].findall(t)))
    hindi = ["हम इस समस्या को हल करने के लिए समीकरणों का उपयोग कर सकते हैं।",
             "आपको नियमित व्यायाम करना चाहिए क्योंकि इससे शरीर की ताकत बढ़ती है।",
             "भारत की राजधानी नई दिल्ली है और यहाँ के लोग मिल-जुलकर रहते हैं।"]
    marathi = ["मी मुंबईतील एक गृहिणी आहे. माझा विमा दावा कसा मांडू?",
               "सेबीच्या नियमनामुळे रचना आता स्पष्ट आणि सुरक्षित झाली आहे."]
    return not any(mar(t) for t in hindi) and all(mar(t) for t in marathi)

json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {DST}: {len(cells)} cells")

# ---------------------------------------------------------------- 5. validate
nb2 = json.load(open(DST, encoding="utf-8"))
bad = []
for i, cc in enumerate(nb2["cells"]):
    if cc["cell_type"] != "code": continue
    try: compile(cell_src(cc), f"cell{i}", "exec")
    except SyntaxError as e: bad.append(f"cell {i} line {e.lineno}: {e.msg} | {cell_src(cc).splitlines()[max(0,e.lineno-1)][:70]}")
assert not bad, "\n  ".join([""] + bad)
allsrc = "\n".join(cell_src(c) for c in nb2["cells"] if c["cell_type"] == "code")
checks = {
    "every code cell compiles":              True,
    # the actual invariant: the cell importing unsloth precedes the first cell
    # importing transformers, or unsloth prints a warning and skips its patches
    "unsloth before transformers":           _order_ok(nb2),
    "v7 data file":                          "train_v7_teacher.jsonl" in allsrc
                                             and "train_v6_teacher" not in allsrc,
    "chains from cpt-hindi":                 'BASE_BRANCH = "cpt-hindi"' in allsrc,
    "~1 epoch of v7":                        "MAX_STEPS     = 3600" in allsrc,
    "GPU guard present":                     "No GPU visible" in allsrc,
    "masking applied and CALLED":            "trainer = train_on_responses_only(" in allsrc,
    "masking hard-fails at zero":            "did not mask anything" in allsrc,
    # word-boundary, not substring: cpt_trainer.train() contains trainer.train()
    "exactly one SFT trainer.train()":       len(re.findall(r"(?<!\w)trainer\.train\(\)", allsrc)) == 1,
    "CPT trainer is actually trained":       len(re.findall(r"cpt_trainer\.train\(\)", allsrc)) == 1,
    "GGUF export present":                   "save_pretrained_gguf" in allsrc,
    "chdir /tmp before GGUF":                'os.chdir("/tmp")' in allsrc,
    "CPT stage enabled":                     "RUN_CPT = True" in allsrc,
    "anchor fully removed":                  "english_anchor" not in allsrc and "prompts_en" not in allsrc,
    "language assertion present":            "_MAR" in allsrc and "Marathi-dominated rows in" in allsrc,
    "no English target rows":                '("hindi", "hinglish")' in allsrc,
    "notebook regex: Hindi करते/सकते not flagged": _regex_probe(nb2),
    "vendor decode settings documented":     "repeat-penalty 1.05" in "".join(nb2["cells"][28]["source"]),
}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("\n  FAILURES BLOCKING THE RUN:", [k for k, v in checks.items() if not v] or "none")
sys.exit(0 if all(checks.values()) else 1)
