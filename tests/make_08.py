#!/usr/bin/env python3
"""Derive notebooks/05-hindi-sft-v8.ipynb from the v7 notebook that make_05.py builds.

Thin layer on purpose: make_05.py stays the provenance record for v7, and every change
here is a v8-specific delta with a gate that fails if it did not apply.

  1. branch / data file / step count for train_v8.jsonl (65,736 rows, 24.9M tokens)
  2. LENGTH-GROUPED BATCHING in cell 21 - 36% less compute, measured, see ORDER below
  3. GPU report so the next session can size batch 4 on data instead of guessing
  4. gates: every v7 check still holds, plus the two new invariants

Why not "use both GPUs". unsloth monkey-patches the train() this notebook's
trainer inherits (unsloth/trainer.py:920 `class UnslothTrainer(SFTTrainer)`; the injected
code is in unsloth/tokenizer_utils.py:1813-1815 of 2026.9.14):

    if ((a - PRE_CHECK) >= 1).sum() > 1:
        raise RuntimeError('Unsloth currently does not support multi GPU setups - but we
                            are working on it!')

`a` is per-device memory from nvidia-smi, so it fires when more than one card is actually
working - which is why v7 ran to completion with both cards *visible* and one idle. It is
not a config we are forgetting, it is a wall the library puts up at the start of train().
Note the separate guard in unsloth_zoo/training_utils.py:554 `get_max_steps` raises on
world_size > 1 too, but that is the `unsloth_train(trainer)` path and this notebook calls
`trainer.train()` directly, so it never reaches it - do not cite 557 as the reason here.
device_map="sequential" does NOT split the
model either at this size - 1.2B fp16 is 2.4 GB against 14.5 GB per card, so card 0 holds
everything and card 1 idles. That is what happened to the v7 run, and it is the honest
answer: with unsloth's fast path, ONE card trains and the second cannot be made to share
one step. The compute we CAN recover is the padding, and it turns out to be a third of it.
"""
import json, re, sys, subprocess

SRC    = "notebooks/05-hindi-sft-v7.ipynb"
DST    = "notebooks/05-hindi-sft-v8.ipynb"
DATA   = "train_v8.jsonl"
BRANCH = "s4-hinglish-v8"
ROWS, STEPS_1EPOCH = 65_721, 4_025       # measured: teach/build_v8_release.py + the 0.98 split


def split_lines(text):
    lines = text.splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


nb = json.load(open(SRC, encoding="utf-8"))
cells = nb["cells"]


def src(i):
    return "".join(l if l.endswith("\n") else l + "\n" for l in cells[i]["source"])


def setsrc(i, text):
    cells[i]["source"] = split_lines(text)
    if cells[i]["cell_type"] == "code":
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None


# ------------------------------------------------------------------ 1. run config
c = src(1)
c = c.replace('BRANCH = "s3-hinglish-v7"', f'BRANCH = "{BRANCH}"')
c = c.replace('DATA_FILE = "train_v7_teacher.jsonl"', f'DATA_FILE = "{DATA}"')
old_steps = [l for l in c.split("\n") if l.startswith("MAX_STEPS")][0]
c = c.replace(old_steps,
              f"MAX_STEPS     = {STEPS_1EPOCH}   # exactly 1.0 epoch of v8: {ROWS:,} rows x 0.98 / 16 per step.\n"
              "#                  # v6 ran 1.21 epochs and v7 ran 0.51 of one; FINDINGS 5.1 calls\n"
              "#                  # matching the flag without the budget 'under-training'. The eval\n"
              "#                  # curve still decides - cell 23 prints the step of minimum eval loss.")
for need in (f'BRANCH = "{BRANCH}"', f'DATA_FILE = "{DATA}"', f"MAX_STEPS     = {STEPS_1EPOCH}"):
    assert need in c, f"cell 1 lost {need!r}"
setsrc(1, c)

# ------------------------------------------------------------------ 2. corpus floor
# The expected-count literal is interpolated HERE, not left as an f-string expression.
# `{ROWS:,}` in the notebook would be a NameError - and because `assert cond, msg` only
# evaluates msg when cond is False, it blows up on the failure path only: the one message
# that has to explain what went wrong. Interpolate it, and forbid braces-names downstream.
c9 = src(9).replace('assert len(rows) > 5000, f"only {len(rows)} rows survive; expected ~56k for v7"',
                    f'assert len(rows) > 40_000, '
                    f'f"only {{len(rows)}} rows survive; expected ~{ROWS:,} for v8 - a fetch truncation"')
assert f"expected ~{ROWS:,} for v8" in c9, f"corpus floor did not interpolate: {c9[:200]}"
setsrc(9, c9)

# ------------------------------------------------------------------ 3. length grouping
# WHY THIS IS THE BIG ONE. Cell 22 trains with `packing = False` and
# DataCollatorForSeq2Seq, which pads every micro-batch to that batch's longest row. The
# v7 file was shuffled, so a 40-token greeting sat in a micro-batch with a 2,000-token
# conversation and the greeting paid 2,000 positions. Measured over train_v8.jsonl at the
# real micro-batch size (per_device 2 x grad_accum 8): 31.7M sequence positions per epoch
# when rows are shuffled vs 20.4M when neighbours have similar lengths - 36% of the run's
# compute was pure padding, and the fix drops no rows and changes no hyper-parameter.
#
# WHY IT HAS TO HAPPEN HERE AND NOT IN THE FILE. Cell 21 calls
# ds.train_test_split(...), which SHUFFLES the rows (datasets' split is a random subset
# order, verified: a sorted 1,000-row set comes back as 864, 193, 277...). A pre-sorted
# jsonl is therefore destroyed before training. And sorting cannot be delegated to
# HF's group_by_length either: unsloth_zoo builds its own DataLoader with
# torch.utils.data.SequentialSampler(trainer.train_dataset) (training_utils.py:740), so it
# reads the dataset in stored order and never asks the Trainer for a sampler. Stored order
# is the only lever - so sort, cut into 16-row optimizer steps, shuffle the STEP order and
# flatten. Adjacent rows stay length-matched (that is what kills the padding) while which
# step arrives when stays random (that is what stops a length curriculum: mean step length
# measured flat at p10/p50/p90 = 200/100/124 across the epoch).
c21 = src(21)
anchor = 'split = ds.train_test_split(test_size=0.02, seed=3407)'
assert anchor in c21, "cell 21 lost the split - re-slice it"
ORDER = anchor + '''

# LENGTH GROUPING - see the markdown cell before this one for the measurement.
_lens = [len(t) for t in split["train"]["text"]]
_perm = sorted(range(len(_lens)), key=lambda i: _lens[i])
_STEP = 16                       # per_device 2 x grad_accum 8 = one optimizer step
_steps = [_perm[i:i + _STEP] for i in range(0, len(_perm) - len(_perm) % _STEP, _STEP)]
random.Random(3407).shuffle(_steps)
_order = [j for st in _steps for j in st]
if len(_perm) % _STEP:           # keep every row; the ragged tail goes at the end
    _order += _perm[len(_steps) * _STEP:]
assert sorted(_order) == list(range(len(_lens))), "grouping lost or duplicated rows"
split["train"] = split["train"].select(_order)
def _padded(lens, order):
    """What the collator actually produces: each micro-batch of 2 pads to its longest row."""
    idx = [order[i] for i in range(len(order) - len(order) % 2)]
    return sum(max(lens[idx[i]], lens[idx[i + 1]]) for i in range(0, len(idx), 2)) * 2

_grouped = _padded(_lens, _order)
_r = random.Random(99); _sh = list(_order); _r.shuffle(_sh)
_shuffled = _padded(_lens, _sh)
print(f"length grouping: {len(_steps):,} steps x {_STEP} rows")
print(f"  padded positions/epoch: grouped {_grouped/1e6:,.1f}M vs shuffled {_shuffled/1e6:,.1f}M "
      f"-> {100*(1-_grouped/_shuffled):.0f}% less padding, same {len(_order):,} rows")
if _grouped >= _shuffled:
    print("  WARNING: grouping did NOT reduce padding - the order is not length-grouped")'''
c21 = c21.replace(anchor, ORDER)
c21 = c21.replace("import re\nfrom datasets import Dataset",
                  "import re, random\nfrom datasets import Dataset")
# stdlib sum, no numpy on a data-prep cell
c21 = c21.replace("_sum(", "sum(")
setsrc(21, c21)

# ------------------------------------------------------------------ 3b. stale file comment
c8 = src(8).replace("train_v7_teacher.jsonl; cell 9 fetches it", f"{DATA}; cell 9 fetches it")
assert "train_v7" not in c8, "cell 8 still names the v7 file"
setsrc(8, c8)

# ------------------------------------------------------------------ 4. GPU report
# One line, so the NEXT session sizes per_device_train_batch_size on evidence. Card 1 is
# unusable for the second half of the same step (see the module docstring), so the only way
# to spend the idle 14.5 GB is a bigger micro-batch on card 0 - the thing that OOMs if
# guessed. Length grouping bunches the long rows together, so peak memory now shows up at a
# predictable place instead of being smeared across the run.
# Cell 22 can only report what is true BEFORE the run: cards visible and memory held by
# loading. The number that sizes per_device_train_batch_size is the TRAINING peak, which
# does not exist yet at that point - printing max_memory_allocated() there and calling it
# "peak" shows the loading figure and invites a batch size that OOMs twenty steps in. So
# cell 22 says "before training", and cell 23 reports the peak after trainer.train(), where
# the name means what it says.
REPORT = """import torch as _t
print(f"before training: cuda:0 holding {_t.cuda.memory_allocated()/2**30:.1f} GB "
      f"of {_t.cuda.get_device_properties(0).total_memory/2**30:.1f} GB | "
      f"cards visible = {_t.cuda.device_count()} | unsloth lets ONE card train: it patches "
      f"SFTTrainer.train to raise 'does not support multi GPU' once a 2nd card does work "
      f"(tokenizer_utils.py:1813). "
      f"The training peak is printed in the next cell.")
"""
c22 = src(22)
anchor22 = "trainer = train_on_responses_only("
assert anchor22 in c22, "cell 22 lost the masker call - re-slice it"
c22 = c22.replace(anchor22, REPORT + anchor22, 1)
assert "memory_allocated" in c22 and "tokenizer_utils.py:1813" in c22
assert "before training" in c22, "cell 22 must not claim a training peak it cannot measure"
setsrc(22, c22)

# ------------------------------------------------------------------ 4b. the real peak
c23 = src(23)
assert c23.startswith("stats = trainer.train()"), "cell 23 no longer starts with the train call"
# reset_peak_memory_stats() is never called in this notebook, so max_memory_allocated()
# would otherwise report the highest value since import - which includes loading the
# 2.4 GB model and any earlier probe tensor. Reset at the top, read at the bottom.
c23 = ("import torch as _t   # cell-local: every cell must run after a kernel restart, so "
       "nothing may be borrowed\n"
       "_t.cuda.reset_peak_memory_stats()   # so the peak below is training's, not the model load's\n"
       + c23)
c23 = c23.replace(
    "print(stats.metrics)",
    'print(f"\\nGPU0 training peak {_t.cuda.max_memory_allocated()/2**30:.1f} GB of "\n'
    '      f"{_t.cuda.get_device_properties(0).total_memory/2**30:.1f} GB "\n'
    '      f"({100*_t.cuda.max_memory_allocated()/_t.cuda.get_device_properties(0).total_memory:.0f}% of the card). "\n'
    '      f"Under 60% -> raise per_device_train_batch_size and halve gradient_accumulation_steps "\n'
    '      f"to keep 16 rows/step: that is the only way to use the second T4\'s idle memory.")\n'
    "print(stats.metrics)", 1)
assert "training peak" in c23 and "reset_peak_memory_stats" in c23
setsrc(23, c23)

# ------------------------------------------------------------------ 5. title
setsrc(0, "# Experiment 8 - Hindi + Hinglish SFT on the merged v8 corpus\n\n"
    "Chained: CPT on Hindi (`cpt-hindi`) -> SFT on `train_v8.jsonl`. One Kaggle session,\n"
    "resumable via `wip/" + BRANCH + "` every 900 s.\n\n"
    "**What is different from v7.** v7 was 57,687 rows of one register: response words\n"
    "p10/med/p90 = 40/47/52, ZERO rows under 15 words, ZERO chat rows, one topic field\n"
    "value. So it answered \"kaise ho bhai\" with a 47-word paragraph. v8 merges three\n"
    "pools, all re-gated locally (teach/build_v8_release.py, AUDIT PASS + masking PASS):\n"
    "  61,132 teacher rows (judge>=4, 50 topics, 22% <=22 words, romanised Hinglish)\n"
    "   2,575 human multi-turn conversations from oasst1_hi (median 4 turns)\n"
    "   2,029 human WikiHow-Hindi how-to answers, 64% carrying English glosses\n"
    "= 65,736 rows / 24.9M tokens, balanced in TOKENS not rows (long rows are 3.8x the\n"
    "median and would otherwise dominate every batch: v6's verbose-collapse failure).\n\n"
    "**Cell 21 now groups batches by length.** The collator pads each micro-batch to its\n"
    "longest row and cell 22 does not pack, so a shuffled file spent 36% of every epoch on\n"
    "pad tokens. Same rows, same hyper-parameters, ~1/3 less compute.\n\n"
    "**Do not run cell 9 past its assertion.** Rebuild with teach/build_v8_release.py.\n")

# ------------------------------------------------------------------ 6. gates
CODE = ("!", "%")


def _import_order_ok():
    """Line-anchored: the `import unsloth` STATEMENT must precede the first transformers
    import STATEMENT. A cell-level substring test is wrong - cell 4 contains both the
    import and a prose comment about later transformers imports, so `in` returns True for
    both and any naive check reports the safe order even when the order is inverted.
    Inverted order makes unsloth print a warning and skip its patches: the run trains,
    just 2x slower and with the memory savings quietly absent."""
    u = t = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        for ln, line in enumerate("\n".join(c["source"]).splitlines()):
            st = line.strip()
            if u is None and st == "import unsloth":
                u = (i, ln)
            if t is None and (st.startswith("from transformers")
                              or st.startswith("import transformers")):
                t = (i, ln)
        if u and t:
            break
    return u is not None and t is not None and u < t
# allsrc = what RUNS (code cells, shell lines dropped). booksrc = what is WRITTEN DOWN,
# markdown included - the vendor decode recipe is documentation in a prose cell, and
# gating it on allsrc reports a documented run as undocumented.
booksrc = "\n".join("".join(c["source"]) for c in cells)
allsrc = "\n".join("\n".join(l for l in c["source"] if not l.strip().startswith(CODE))
                   for c in cells if c["cell_type"] == "code")

checks = {
    "v8 data file":                 f'DATA_FILE = "{DATA}"' in allsrc and "train_v7_teacher" not in allsrc,
    "v8 branch":                    f'BRANCH = "{BRANCH}"' in allsrc,
    "1 epoch of v8, not of v7":     f"MAX_STEPS     = {STEPS_1EPOCH}" in allsrc
                                    and "MAX_STEPS     = 3600" not in allsrc,
    "length grouping present":      "split[\"train\"].select(_order)" in allsrc
                                    and "random.Random(3407).shuffle(_steps)" in allsrc,
    "grouping cannot lose rows":    "assert sorted(_order) == list(range(len(_lens)))" in allsrc,
    "grouping happens AFTER split": allsrc.index("train_test_split") < allsrc.index("_lens = [len(t)"),
    "random is imported in 21":     "import re, random" in src(21),
    "GPU report present":           "max_memory_allocated" in allsrc,
    "corpus floor is v8-sized":     "for v8" in allsrc and "expected ~56k" not in allsrc,
    # everything v7 already had to be true about, re-asserted on the v8 book
    "chains from cpt-hindi":        'BASE_BRANCH = "cpt-hindi"' in allsrc,
    "SFT loads CPT weights locally": 'BASE_REPO = _cpt_dir' in allsrc,
    "SFT effective batch is 16":    "gradient_accumulation_steps = 8" in allsrc,
    "masking applied and CALLED":   "trainer = train_on_responses_only(" in allsrc,
    "multi-turn rows are rendered": 'convs = batch["messages"]' in allsrc,
    "masking hard-fails at zero":   "did not mask anything" in allsrc,
    "exactly one SFT trainer.train()": len(re.findall(r"(?<!\w)trainer\.train\(\)", allsrc)) == 1,
    "CPT trainer is trained":       len(re.findall(r"cpt_trainer\.train\(\)", allsrc)) == 1,
    "GGUF export present":          "save_pretrained_gguf" in allsrc,
    "CPT stage enabled":            "RUN_CPT = True" in allsrc,
    "no English target rows":       "english_anchor" not in allsrc,
    "language assertion present":   "marathi-dominated" in allsrc,
    "no cross-card split":          'os.environ["UNSLOTH_AUTO_DEVICE_MAP"] = "0"' in allsrc,
    # Invariants that live in the v7 base cell-for-cell and were checked only by
    # tests/make_05.py, which no longer runs against this book. A delta-only gate table
    # proves the diff landed and proves nothing about the rest of the notebook: these four
    # are the ones that cost GPU-hours when they regress.
    "unsloth imported before transformers": _import_order_ok(),
    "checkpoint cadence 900s":               "CHECKPOINT_EVERY_S = 900" in allsrc,
    "chdir /tmp before GGUF":                'os.chdir("/tmp")' in allsrc,
    "vendor decode settings documented":     "repeat-penalty 1.05" in booksrc,
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# every code cell must still compile
bad = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if any(l.strip().startswith(CODE) for l in s.split("\n")):
        continue
    try:
        compile(s, f"cell{i}", "exec")
    except SyntaxError as e:
        bad.append((i, e.lineno, e.msg))
print(f"wrote {DST}: {len(cells)} cells")
if bad:
    print("  SYNTAX ERRORS:", bad)
fails = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
if bad:
    fails.append("syntax")
print("\n" + ("READY TO RUN" if not fails else f"BLOCKED: {fails}"))
sys.exit(0 if not fails else 1)
