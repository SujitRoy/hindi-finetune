#!/usr/bin/env python3
"""Derive notebooks/05-hindi-sft-v8.ipynb from the v7 notebook that make_05.py builds.

Thin layer on purpose: make_05.py stays the provenance record for v7, and every change
here is a v8-specific delta with a gate that fails if it did not apply.

  1. branch / data file / step count for train_v8.jsonl (65,736 rows, 24.9M tokens)
  2. LENGTH-GROUPED BATCHING - 36% less compute, measured, see the writer cell
  3. BOTH T4s TRAIN. v7 ran one process on one card and the second 15 GB T4 idled.
     The old note blamed unsloth's `does not support multi GPU` raise at
     tokenizer_utils.py:1813. That raise is dead code in the installed wheel:
     `patch_sft_trainer_tokenizer()` is defined but has NO call site in
     unsloth-2026.9.14, so it is never installed on `SFTTrainer.train`. The supported
     path (unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) is DDP via torchrun,
     and unsloth's own planner pins each rank to its card once WORLD_SIZE > 1.
     The notebook now writes notebooks/train_ddp.py into /kaggle/working and launches
     it with `python -m torch.distributed.run --standalone --nproc_per_node=2`.
  4. gates: every v7 check still holds, plus the multi-GPU invariants.
"""
import json, re, sys

SRC    = "notebooks/05-hindi-sft-v7.ipynb"
DST    = "notebooks/05-hindi-sft-v8.ipynb"
SCRIPT = "notebooks/train_ddp.py"
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


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": split_lines(text)}
    if kind == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


# ------------------------------------------------------------------ 1. run config
c = src(1)
c = c.replace('BRANCH = "s3-hinglish-v7"', f'BRANCH = "{BRANCH}"')
c = c.replace('DATA_FILE = "train_v7_teacher.jsonl"', f'DATA_FILE = "{DATA}"')
old_steps = [l for l in c.split("\n") if l.startswith("MAX_STEPS")][0]
c = c.replace(old_steps,
              f"MAX_STEPS     = {STEPS_1EPOCH}   # exactly 1.0 epoch of v8: {ROWS:,} rows x 0.98 / 16 per step.\\n"
              "#                  # v6 ran 1.21 epochs and v7 ran 0.51 of one; FINDINGS 5.1 calls\\n"
              "#                  # matching the flag without the budget 'under-training'. The eval\\n"
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

# ------------------------------------------------------------------ 3. stale file comment
c8 = src(8).replace("train_v7_teacher.jsonl; cell 9 fetches it", f"{DATA}; cell 9 fetches it")
assert "train_v7" not in c8, "cell 8 still names the v7 file"
setsrc(8, c8)

# ------------------------------------------------------------------ 4. writer cell
# WHY LENGTH GROUPING IS HERE AND NOT IN THE SCRIPT. Cell 21 used to render the
# conversations and group the batches, and the old trainer cell trained on that order.
# The script must train on the SAME order, so the rendering + grouping is done once,
# here, and saved; the script only loads what was saved. The collator pads every
# micro-batch to its own longest row, and cell 22 does not pack, so a shuffled file
# spent 36% of every epoch on pads (31.7M padded positions vs 20.4M grouped, measured
# over train_v8.jsonl at micro-batch 2). Grouping drops no rows and changes no
# hyper-parameter. It cannot be delegated to HF's group_by_length either:
# unsloth_zoo builds its own DataLoader with a SequentialSampler over the stored order,
# so the stored order is the only lever. Sort, cut into 16-row optimizer steps, shuffle
# the STEP order, flatten - adjacent rows stay length-matched, which step arrives when
# stays random.
WRITER = '''import json, random
from datasets import Dataset

# ---- run config for train_ddp.py. This cell is the one place a run is set.
RUN_CPT    = True                 # Liquid stage 1: install Hindi before SFT
CPT_STEPS  = 1500                 # 1500 steps x 16 rows = 24,000 docs, ~2.5 h on 2 cards
CPT_SEQ    = 1024                 # measured: median 528, p90 987 tokens
CPT_LR     = 5e-5                 # Liquid cpt_text_completion / cpt_translation
CPT_EMB_LR = 1e-5                 # "2 to 10x smaller than learning_rate"
CPT_R      = 128                  # "Add for continual pretraining" uses 128; SFT uses 16
CPT_ALPHA  = 32
CPT_FILE   = "cpt_sample.jsonl"
CPT_BS, CPT_ACC = 1, 8            # 1 x 8 x 2 GPUs = 16 rows/step
SFT_BS, SFT_ACC = 2, 4            # 2 x 4 x 2 GPUs = 16 rows/step, same budget as v7
SFT_BASE   = "/kaggle/working/cpt-merged" if RUN_CPT else BASE_REPO

# ---- SFT split: render every conversation, then group batches by length. The
# script only loads what this cell saves, so this order is what training sees.
def _pair(r):
    return [{"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]}]

rows = [{**r, "messages": r.get("messages") or _pair(r)} for r in rows]

def to_text(batch):
    convs = batch["messages"]
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False,
                                                   add_generation_prompt=False)
                            .removeprefix(tokenizer.bos_token or "")
                     for c in convs]}

ds = Dataset.from_list(rows).map(to_text, batched=True, remove_columns=list(rows[0].keys()))
split = ds.train_test_split(test_size=0.02, seed=3407)

_lens = [len(t) for t in split["train"]["text"]]
_perm = sorted(range(len(_lens)), key=lambda i: _lens[i])
_STEP = 16                       # per_device 2 x grad_accum 4 x 2 GPUs = one optimizer step
_steps = [_perm[i:i + _STEP] for i in range(0, len(_perm) - len(_perm) % _STEP, _STEP)]
random.Random(3407).shuffle(_steps)
_order = [j for st in _steps for j in st]
if len(_perm) % _STEP:           # keep every row; the ragged tail goes at the end
    _order += _perm[len(_steps) * _STEP:]
assert sorted(_order) == list(range(len(_lens))), "grouping lost or duplicated rows"
split["train"] = split["train"].select(_order)

def _padded(lens, order):
    idx = [order[i] for i in range(len(order) - len(order) % 2)]
    return sum(max(lens[idx[i]], lens[idx[i + 1]]) for i in range(0, len(idx), 2)) * 2

_grouped = _padded(_lens, _order)
_r = random.Random(99); _sh = list(_order); _r.shuffle(_sh)
_shuffled = _padded(_lens, _sh)
print(f"length grouping: {len(_steps):,} steps x {_STEP} rows")
print(f"  padded positions/epoch: grouped {_grouped/1e6:,.1f}M vs shuffled {_shuffled/1e6:,.1f}M "
      f"-> {100*(1-_grouped/_shuffled):.0f}% less padding, same {len(_order):,} rows")
if _grouped >= _shuffled:
    print("  WARNING: grouping did NOT reduce padding - the order is not length-grouped")
print(f"SFT split: train {len(split['train']):,} eval {len(split['test']):,} rows")
assert len(split["train"]) > 40_000, "the split is far smaller than v8 - a fetch truncation"
split.save_to_disk("/kaggle/working/sft_split")

# ---- the script reads this, so a value can never drift between notebook and training
CFG = {
    "REPO": REPO, "BRANCH": BRANCH, "WIP": WIP,
    "BASE_REPO": BASE_REPO, "SFT_BASE": SFT_BASE,
    "DATA_REPO": DATA_REPO, "TOK_DIR": TOK_DIR,
    "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH, "DTYPE": "float16",
    "MAX_STEPS": MAX_STEPS, "LEARNING_RATE": LEARNING_RATE,
    "CHECKPOINT_EVERY_S": CHECKPOINT_EVERY_S,
    "RUN_CPT": RUN_CPT, "CPT_STEPS": CPT_STEPS, "CPT_SEQ": CPT_SEQ,
    "CPT_LR": CPT_LR, "CPT_EMB_LR": CPT_EMB_LR, "CPT_R": CPT_R, "CPT_ALPHA": CPT_ALPHA,
    "CPT_FILE": CPT_FILE, "CPT_BS": CPT_BS, "CPT_ACC": CPT_ACC,
    "SFT_BS": SFT_BS, "SFT_ACC": SFT_ACC, "SFT_SPLIT": "/kaggle/working/sft_split",
    "WANDB_ON": WANDB_ON, "WANDB_PROJECT": WANDB_PROJECT,
}
json.dump(CFG, open("/kaggle/working/ddp_config.json", "w"), indent=1)
print("ddp_config.json written - RUN_CPT =", RUN_CPT)
'''

LAUNCH = '''import os, subprocess, sys   # cell-local: every cell must run after a kernel restart

SCRIPT = "/kaggle/working/train_ddp.py"

def _launch(stage, n_gpu=2):
    """One process per T4. torchrun sets LOCAL_RANK/WORLD_SIZE on each, which is
    what unsloth's planner reads to pin the model to that rank's own card."""
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={n_gpu}", SCRIPT, "--stage", stage]
    print("==>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=os.environ.copy())

if RUN_CPT:
    _launch("cpt")        # continued pretraining on both T4s, merged to cpt-hindi
_launch("sft")            # SFT on both T4s, merged -> GGUF -> BRANCH
print("training done on 2 x T4; rank 0 pushed the merged GGUF + adapter")
'''

script_text = open(SCRIPT, encoding="utf-8").read().rstrip("\n")
assert "'''" not in script_text, "train_ddp.py contains ''' and cannot be embedded in the writer"
assert not any(l.strip().startswith(("!", "%")) for l in script_text.split("\n")), \
    "train_ddp.py has a shell/magic line"

MD_RUN = '''## 3. Train on BOTH T4s - DDP via `torchrun`

The notebook used to run one process on one card, and the second 15 GB T4 sat idle.
That was blamed on a hard block in Unsloth, but `patch_sft_trainer_tokenizer()` - the
only place the `does not support multi GPU` raise lives - has **no call site** in the
installed `unsloth-2026.9.14` wheel, so it is never installed on `SFTTrainer.train`.
The documented path,
<https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth>, is Distributed Data
Parallel launched with `torchrun`, and that is what the next three cells do:

1. **the config cell** renders every conversation, length-groups the batches, saves the
   split and `ddp_config.json`, and exposes `RUN_CPT` / batch sizes.
2. **the `%%writefile` cell** writes `train_ddp.py` - the whole training program - to
   `/kaggle/working`.
3. **the launch cell** runs it under
   `python -m torch.distributed.run --standalone --nproc_per_node=2`: one process per T4.

Each rank loads the model on its own card (Unsloth reads `LOCAL_RANK`/`WORLD_SIZE` and
pins the device), trains on its shard of the data, and DDP all-reduces the LoRA
gradients once per optimizer step. Two T4s therefore double rows/step at the **same
effective batch of 16** - no hyper-parameter changes. Rank 0 alone owns the Hub: the
wall-clock checkpoints, the merged `cpt-hindi` push, and the final merge -> GGUF.

Model targets, `use_rslora` and the response masking are all inside `train_ddp.py`,
so there is one copy of the training logic rather than a notebook cell and a script
that can drift apart.'''

MD_ART = '''## 6. Merge -> GGUF -> Hub (done by the training run)

The `train_ddp.py` launch ends on **rank 0** by merging the adapter, converting to
`q4_k_m` GGUF and pushing both to `BRANCH` - the work that used to be a separate cell.
The branch is the only thing that survives a Kaggle VM reset, so before ending the
session confirm it landed:

    https://huggingface.co/<you>/LFM2.5-1.2B-Instruct-HI-Uncensored/tree/s4-hinglish-v8

The five-minute `ckpt-*` adapters on `WIP` are the resume points if the session dies.'''

# ------------------------------------------------------------------ 5. rebuild cells
new_cells = []
new_cells += cells[0:15]                     # config .. local tokenizer (0..14)
new_cells += [cells[15]]                     # CPT markdown
new_cells += [cells[16]]                     # W&B cell
new_cells += [cell("markdown", MD_RUN)]
new_cells += [cell("code", WRITER)]
new_cells += [cell("code", "%%writefile /kaggle/working/train_ddp.py\n" + script_text)]
new_cells += [cell("code", LAUNCH)]
new_cells += [cells[24], cells[25]]          # licence markdown + code
new_cells += [cell("markdown", MD_ART)]
new_cells += [cells[28]]                     # server markdown
nb["cells"] = cells = new_cells

# ------------------------------------------------------------------ 6. title
setsrc(0, "# Experiment 8 - Hindi + Hinglish SFT on the merged v8 corpus\n\n"
    "Chained: CPT on Hindi (`cpt-hindi`) -> SFT on `train_v8.jsonl`. One Kaggle session,\n"
    "resumable via `wip/" + BRANCH + "` every 900 s.\n\n"
    "**Both T4s train.** One process per GPU under `torch.distributed.run`; rank 0 owns\n"
    "the Hub. The old one-card run was not a missing flag - the Unsloth multi-GPU raise\n"
    "is dead code in the installed wheel. See the DDP cell for the citation.\n\n"
    "**What is different from v7.** v7 was 57,687 rows of one register: response words\n"
    "p10/med/p90 = 40/47/52, ZERO rows under 15 words, ZERO chat rows, one topic field\n"
    "value. So it answered \"kaise ho bhai\" with a 47-word paragraph. v8 merges three\n"
    "pools, all re-gated locally (teach/build_v8_release.py, AUDIT PASS + masking PASS):\n"
    "  61,132 teacher rows (judge>=4, 50 topics, 22% <=22 words, romanised Hinglish)\n"
    "   2,575 human multi-turn conversations from oasst1_hi (median 4 turns)\n"
    "   2,029 human WikiHow-Hindi how-to answers, 64% carrying English glosses\n"
    "= 65,736 rows / 24.9M tokens, balanced in TOKENS not rows (long rows are 3.8x the\n"
    "median and would otherwise dominate every batch: v6's verbose-collapse failure).\n\n"
    "**Batches are grouped by length.** The collator pads each micro-batch to its\n"
    "longest row and the trainer does not pack, so a shuffled file spent 36% of every\n"
    "epoch on pad tokens. Same rows, same hyper-parameters, ~1/3 less compute.\n\n"
    "**Do not run the data cell past its assertion.** Rebuild with teach/build_v8_release.py.\n")

# ------------------------------------------------------------------ 7. gates
CODE = ("!", "%")


def _import_order_ok():
    """Line-anchored: the `import unsloth` STATEMENT must precede the first transformers
    import STATEMENT. A cell-level substring test is wrong - cell 4 contains both the
    import and a prose comment about later transformers imports. Inverted order makes
    unsloth print a warning and skip its patches: the run trains, just 2x slower."""
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


booksrc = "\n".join("".join(c["source"]) for c in cells)
allsrc = "\n".join("\n".join(l for l in c["source"] if not l.strip().startswith(CODE))
                   for c in cells if c["cell_type"] == "code")

checks = {
    "v8 data file":                 f'DATA_FILE = "{DATA}"' in allsrc and "train_v7_teacher" not in allsrc,
    "v8 branch":                    f'BRANCH = "{BRANCH}"' in allsrc,
    "1 epoch of v8, not of v7":     f"MAX_STEPS     = {STEPS_1EPOCH}" in allsrc
                                    and "MAX_STEPS     = 3600" not in allsrc,
    "length grouping present":      'split["train"].select(_order)' in allsrc
                                    and "random.Random(3407).shuffle(_steps)" in allsrc,
    "grouping cannot lose rows":    "assert sorted(_order) == list(range(len(_lens)))" in allsrc,
    "grouping happens AFTER split": allsrc.index("train_test_split") < allsrc.index("_lens = [len(t)"),
    "SFT split saved for the script": 'save_to_disk("/kaggle/working/sft_split")' in allsrc,
    "corpus floor is v8-sized":     "for v8" in allsrc and "expected ~56k" not in allsrc,
    # --- the multi-GPU delta and its invariants ---
    "DDP script is written out":    "%%writefile /kaggle/working/train_ddp.py" in booksrc,
    "DDP launched with torchrun":   "torch.distributed.run" in allsrc,
    "one process per T4":           "--nproc_per_node=2" in allsrc or "nproc_per_node={n_gpu}" in allsrc,
    "both ranks train, not split":  "LOCAL_RANK" in allsrc and "WORLD_SIZE" in allsrc,
    "rank 0 owns the Hub":          "is_world_process_zero" in allsrc,
    # The correction has to be documented, not just asserted: the script cites the
    # dead raise, so a bare substring ban would fail on its own explanation.
    "no unsloth one-card claim":    "lets ONE card train" not in booksrc
                                    and "patch_sft_trainer_tokenizer()" in booksrc
                                    and "--nproc_per_node" in booksrc,
    "SFT effective batch is 16":    "SFT_BS, SFT_ACC = 2, 4" in allsrc,
    "CPT effective batch is 16":    "CPT_BS, CPT_ACC = 1, 8" in allsrc,
    "GPU report present":           "max_memory_allocated" in allsrc,
    # --- everything v7 already had to be true about, re-asserted on the v8 book ---
    "chains from cpt-hindi":        'BASE_BRANCH = "cpt-hindi"' in allsrc,
    "SFT merges the CPT weights":   'save_pretrained_merged(CFG["SFT_BASE"]' in allsrc
                                    and 'CFG["SFT_BASE"]' in allsrc,
    "masking applied and CALLED":   "trainer = train_on_responses_only(" in allsrc,
    "multi-turn rows are rendered": 'convs = batch["messages"]' in allsrc,
    "masking hard-fails at zero":   "did not mask anything" in allsrc,
    "exactly one SFT trainer.train()": len(re.findall(r"(?<!\w)trainer\.train\(\)", allsrc)) == 1,
    "CPT trainer is trained":       len(re.findall(r"cpt_trainer\.train\(\)", allsrc)) == 1,
    "GGUF export present":          "save_pretrained_gguf" in allsrc,
    "CPT stage enabled":            "RUN_CPT    = True" in allsrc,
    "no English target rows":       "english_anchor" not in allsrc,
    "language assertion present":   "marathi-dominated" in allsrc,
    "unsloth imported before transformers": _import_order_ok(),
    "checkpoint cadence 900s":      "CHECKPOINT_EVERY_S = 900" in allsrc,
    "chdir /tmp before GGUF":       'os.chdir("/tmp")' in allsrc,
    "vendor decode settings documented": "repeat-penalty 1.05" in booksrc,
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
# and the script the notebook writes must compile as a standalone program
try:
    compile(script_text, SCRIPT, "exec")
except SyntaxError as e:
    bad.append((SCRIPT, e.lineno, e.msg))

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
