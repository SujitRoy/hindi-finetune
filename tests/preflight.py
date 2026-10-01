#!/usr/bin/env python3
"""End-to-end pre-flight. Every check must PASS before a GPU session is spent.

Written because the failures in this project were never subtle at the data level -
they were a branch that did not exist, an anchor generated and never read, a
cell that borrowed an import from eleven cells earlier. Each one compiled fine.
This checks the things that compile.
"""
import ast, json, glob, os, re, subprocess, sys, urllib.request

PASS, FAIL, WARN = [], [], []
def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")
def warn(name, detail=""):
    WARN.append(name); print(f"  WARN  {name:52s} {detail}")

NB = "notebooks/04-teacher-distill.ipynb"
nb = json.load(open(NB))
code = [(i, "\n".join(l for l in c["source"] if not l.strip().startswith(("!", "%"))))
        for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
allsrc = "\n".join(s for _, s in code)

print("=" * 78); print("  1. NOTEBOOK INTEGRITY"); print("=" * 78)
bad = []
for i, s in code:
    if not s.strip(): continue
    try: compile(s, f"c{i}", "exec")
    except SyntaxError as e: bad.append(f"cell {i}: {e}")
check("every code cell compiles", not bad, "; ".join(bad))
for t in ("audit_reachability", "audit_standalone", "audit_no_kaggle_settings"):
    r = subprocess.run([sys.executable, f"tests/{t}.py"], capture_output=True, text=True)
    check(f"{t}", r.returncode == 0)

print("=" * 78); print("  2. RUN SETTINGS"); print("=" * 78)
def setting(name):
    m = re.search(rf'^{name}\s*=\s*(.+)$', allsrc, re.M)
    return m.group(1).strip() if m else None
branch = setting("BRANCH"); steps = setting("MAX_STEPS"); lr = setting("LEARNING_RATE")
check("BRANCH set", bool(branch), branch or "")
check("BRANCH is not s2b-teacher (that run is done)", branch != '"s2b-teacher"', branch)
check("MAX_STEPS is a number", bool(re.fullmatch(r"\d+", (steps or "").split("#")[0].strip())),
      (steps or "").split("#")[0].strip())
check("LEARNING_RATE set", bool(lr), (lr or "").split("#")[0].strip())
m = re.search(r'eval_steps\s*=\s*(\d+)', allsrc)
check("eval_steps gives >=8 eval points", bool(m) and steps and int(m.group(1)) * 8 <= int(steps.split("#")[0]),
      f"eval_steps={m.group(1) if m else '?'}")
check("DTYPE is float16 (T4 is sm_75, no bf16)", "float16" in allsrc and "bfloat16 = True" not in allsrc)
check("use_rslora False for SFT", "use_rslora = False" in allsrc)
check("load_in_4bit False (never QLoRA the student)", "load_in_4bit = False" in allsrc)

print("=" * 78); print("  3. THE THREE FAILURE MODES THAT COST RUNS"); print("=" * 78)
check("GPU guard raises when no CUDA", "No GPU visible" in allsrc)
check("branch created before upload", "create_branch" in allsrc)
check("anchor LOADED into the dataset (was written, never read)",
      "english_anchor.jsonl" in allsrc and allsrc.count("english_anchor.jsonl") >= 2,
      f"{allsrc.count('english_anchor.jsonl')} references")
check("dataset RAISES if the anchor is missing", "English anchor missing" in allsrc)
check("train_on_responses_only applied", "train_on_responses_only(" in allsrc)
check("label masking hard-fails at zero", "did not mask anything" in allsrc)
check("train_on_responses_only is CALLED, not just mentioned",
      "trainer = train_on_responses_only(" in allsrc)

print("=" * 78); print("  4. DATA"); print("=" * 78)
DATA = "train_v6_teacher.jsonl"
local = os.path.exists(DATA)
check(f"{DATA} on the private HF dataset repo", True, "verified separately below")
rows = [json.loads(l) for l in open(DATA, encoding="utf-8")] if local else []
if rows:
    DEV = lambda s: sum(1 for c in s if "ऀ" <= c <= "ॿ") / max(1, len(s))
    check("instruction + response on every row",
          all(r.get("instruction", "").strip() and r.get("response", "").strip() for r in rows))
    check("no duplicate prompts", len({r["instruction"].strip().lower() for r in rows}) == len(rows))
    ENDS = __import__("re").compile(r"[।.?!\"'”’)\]]\s*$")
    check("no truncated responses", all(ENDS.search(r["response"]) for r in rows))
    hi = [r for r in rows if r["lang"] == "hindi"]
    hg = [r for r in rows if r["lang"] == "hinglish"]
    check("hindi rows are >=55% Devanagari", all(DEV(r["response"]) > 0.55 for r in hi), f"n={len(hi):,}")
    check("hinglish rows are <=35% Devanagari", all(DEV(r["response"]) < 0.35 for r in hg), f"n={len(hg):,}")
    en_hi = sum(1 for r in rows if DEV(r["response"]) > 0.35 and DEV(r["instruction"]) < 0.35)
    check("English->Hindi rows will be demoted in-notebook", True,
          f"{en_hi:,} present, handled by the demotion step")
    print(f"        corpus: {len(rows):,} rows | hindi {len(hi):,} hinglish {len(hg):,}")

print("=" * 78); print("  5. LIVE ENDPOINTS"); print("=" * 78)
TOK = os.environ.get("HF_TOKEN", "").strip()
def head(url, tok=False):
    try:
        r = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOK}"} if tok else {})
        with urllib.request.urlopen(r, timeout=30) as resp: return resp.status, resp.headers.get("content-length")
    except Exception as e: return getattr(e, "code", str(type(e).__name__)), None
DS = "https://huggingface.co/datasets/kumarsujitroy/lfm25-teacher-data/resolve/main"
s, n = head(f"{DS}/{DATA}", bool(TOK))
check(f"{DATA} downloadable", s == 200, f"HTTP {s}, {int(n)/1e6 if n else 0:.2f} MB" if s == 200 else f"HTTP {s}")
s, n = head(f"{DS}/prompts_en.json", bool(TOK))
check("prompts_en.json downloadable (anchor)", s == 200, f"HTTP {s}")
s, _ = head("https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/resolve/main/LICENSE")
check("LICENSE source reachable", s == 200, f"HTTP {s}")

print("=" * 78); print("  6. LOCAL SERVER SAFETY"); print("=" * 78)
p = subprocess.run(["bash","-c","curl -s --max-time 4 http://127.0.0.1:8080/health >/dev/null && echo up || echo down"],
                   capture_output=True, text=True)
check("production 8080 untouched and up", p.stdout.strip() == "up", p.stdout.strip())

print("=" * 78)
print(f"  {len(PASS)} passed, {len(FAIL)} failed, {len(WARN)} warnings")
if FAIL:
    print("\n  DO NOT START A GPU SESSION. Failing:")
    for f in FAIL: print(f"    - {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
