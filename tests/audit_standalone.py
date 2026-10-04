#!/usr/bin/env python3
"""Every code cell must run in a namespace where only the cells above it have run.

A cell that borrows an import from an earlier cell dies the moment it is re-run
alone - which is exactly what happens after a restart, and it cost a full
training run once already: the LICENSE cell used `os` without importing it.
Names the cell assigns itself do not count as missing, and `from x import y`
counts as an import.
"""
import ast, builtins, json, glob, sys

def defined_and_used(tree):
    imported, assigned = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.asname or a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported |= {a.asname or a.name for a in n.names}
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            assigned.add(n.id)
        elif isinstance(n, ast.arg):
            assigned.add(n.arg)
        elif isinstance(n, (ast.comprehension,)):
            pass
        elif isinstance(n, ast.ExceptHandler) and n.name:
            assigned.add(n.name)
    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - imported - assigned - set(dir(builtins))

# the notebook's shared config, produced by the config cell at index 1 and
# consumed by everything downstream. These are legitimately cell-order dependent.
SHARED = {
    # produced by the config cell (index 1)
    "REPO", "BRANCH", "BASE_REPO", "BASE_REV", "WIP", "DATA_FILE", "MAX_STEPS",
    "LEARNING_RATE", "CHECKPOINT_EVERY_S", "MAX_SEQ_LENGTH", "DTYPE", "TOK_DIR",
    "TOKENIZER_REPO", "HF_USER", "OUT_REPO", "SESSION",
    # produced by the setup / data / model cells
    "model", "tokenizer", "trainer", "dataset", "split", "rows", "stats",
    "TimeCheckpoint", "FastLanguageModel", "UnslothTrainer",
    "UnslothTrainingArguments", "Dataset", "load_dataset", "AutoTokenizer",
    "HfApi", "create_repo", "upload_file", "upload_folder", "login", "whoami",
    "TrainerCallback", "tokens_decoded_metric", "per_device_train_batch_size",
    # produced by the wandb cell inserted before the trainer
    "WANDB_ON", "WANDB_PROJECT", "DATA_REPO",
    # produced by the v8 config cell and read by the DDP launch cell
    "RUN_CPT", "CFG",
    "gradient_accumulation_steps", "eval_dataset", "text", "_ratio", "_hist",
}

bad = 0
for path in sorted(glob.glob("notebooks/*.ipynb")) + ["hinglish-sft-lfm25-1.2b.ipynb"]:
    nb = json.load(open(path))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code": continue
        src = "\n".join(l for l in cell["source"] if not l.strip().startswith(("!", "%")))
        if not src.strip(): continue
        try: tree = ast.parse(src)
        except SyntaxError: continue
        miss = defined_and_used(tree) - SHARED
        if miss:
            print(f"{path} cell {i}: missing {sorted(miss)}")
            bad += 1
print("=" * 70)
print(f"  FAIL - {bad} cell(s) depend on an import they do not make"
      if bad else "  PASS - every cell imports what it uses")
sys.exit(1 if bad else 0)
