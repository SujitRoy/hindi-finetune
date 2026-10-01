#!/usr/bin/env python3
"""Every notebook must compile AND its load-bearing calls must be reachable.

Compiling is not enough. A de-indent once re-parented all 85 lines of the CPT
cell under `if isinstance(cpt_model, PeftModelForCausalLM)`, which is False for
a freshly loaded model. The cell compiled, it printed nothing, and it would have
published 300 steps of nothing. This walks the AST and reports the guard each
load-bearing call sits behind, so a `train()` buried under a False test is
visible before a GPU session is spent on it.
"""
import ast, json, glob, sys

LOAD_BEARING = {"train", "get_peft_model", "save_pretrained_merged",
                "push_to_hub", "push_to_hub_gguf", "upload_folder", "generate"}
# guards that are genuinely False for this code and would therefore hide work
ALWAYS_FALSE = ("PeftModelForCausalLM", "RUN_CPT) ==", "is None) and")

def walk(node, guards, out):
    for c in ast.iter_child_nodes(node):
        g = guards
        if isinstance(c, ast.Call):
            f = c.func
            nm = getattr(f, "attr", None) or getattr(f, "id", None)
            if nm in LOAD_BEARING:
                out.append((nm, guards))
        if isinstance(c, ast.If):
            g = guards + [ast.unparse(c.test)[:70]]
        walk(c, g, out)

bad = 0
for path in sorted(glob.glob("notebooks/*.ipynb")) + ["hinglish-sft-lfm25-1.2b.ipynb"]:
    nb = json.load(open(path))
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(l for l in cell["source"] if not l.strip().startswith(("!", "%")))
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"{path} cell {i}: SYNTAX {e}"); bad += 1; continue
        hits = []
        walk(tree, [], hits)
        for nm, guards in hits:
            for g in guards:
                if any(k in g for k in ALWAYS_FALSE):
                    print(f"{path} cell {i}: {nm}() hidden behind `if {g}`")
                    bad += 1
print("=" * 70)
print("  FAIL - work is unreachable" if bad else "  PASS - every load-bearing call is reachable")
sys.exit(1 if bad else 0)
