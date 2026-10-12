# ============================================================================
# PASTE THIS INTO A NEW CELL once the training cell has finished.
#
# Why it exists: the run that is training right now (session 2026-10-12) pushed
# nothing for the whole SFT stage, because every upload_folder carried a README
# whose YAML header said
#     base_model: /tmp/cpt-merged
# and the Hub rejects a base_model that is not a model id:
#     ValueError: Invalid metadata in README.md.
# unsloth builds that header from model.config._name_or_path, and the SFT model
# was loaded from a local directory. The automatic publish cells at the end of
# the notebook go through unsloth's own push_to_hub / push_to_hub_gguf, which
# read the same field, so they will fail the same way. This cell sidesteps them:
# save locally, repair the card, upload_folder - the pattern the CPT cell in the
# notebook already uses successfully.
#
# Self-contained on purpose: the kernel that is training has the OLD version of
# the checkpoint cell, so hub_safe_card does not exist there yet.
#
# Idempotent - safe to run twice, and each stage reports what it actually did.
# Assumes it runs in the SAME kernel as the training cell (it needs `model`,
# `tokenizer` and `trainer` in memory).
# ============================================================================

REPO   = globals().get("REPO",   "kumarsujitroy/LFM2.5-1.2B-Instruct-HI-Uncensored")
BRANCH = globals().get("BRANCH", "s4-hinglish-v8")
WIP    = globals().get("WIP",    f"wip/{BRANCH}")
UPSTREAM_ID = "zaakirio/LFM2.5-1.2B-Instruct-Uncensored"

import os, shutil, time
from huggingface_hub import upload_folder, create_branch, model_info
from huggingface_hub.utils import RevisionNotFoundError


def hub_safe_card(directory, fallback_id = UPSTREAM_ID, repo_type = "model"):
    """Make the README in `directory` acceptable to the Hub, or drop its YAML header.

    The Hub validates the YAML header of every README it receives and rejects a
    base_model that is not a model id. Never delete the file: the prose and every
    weight file stay, only the metadata goes.
    """
    import re as _re
    p = os.path.join(directory, "README.md")
    if not os.path.exists(p):
        return "no README.md"
    _txt = open(p, encoding = "utf-8").read()

    def _fix(m):
        v = m.group(2)
        return m.group(0) if ("/" in v and not v.startswith("/")) else f"{m.group(1)}{fallback_id}"
    _new = _re.sub(r"(?m)^(\s*base_model\s*:\s*)(\S+)\s*$", _fix, _txt)
    if _new == _txt:
        return "card already valid"
    open(p, "w", encoding = "utf-8").write(_new)
    from huggingface_hub import RepoCard
    try:
        RepoCard.load(p).validate(repo_type = repo_type)
        return f"repaired base_model -> {fallback_id}"
    except Exception as e:
        _body = _re.sub(r"\A---\n.*?\n---\n", "", _new, flags = _re.S)
        open(p, "w", encoding = "utf-8").write(_body)
        return f"stripped YAML header ({type(e).__name__})"


def ensure_branch(repo, rev, source = None):
    try:
        model_info(repo, revision = rev)
        return f"branch {rev} exists"
    except RevisionNotFoundError:
        create_branch(repo, branch = rev, revision = source, exist_ok = True)
        return f"branch {rev} created"


def step_now():
    """Last global_step this trainer reached, without assuming log_history exists."""
    try:
        return int(trainer.state.global_step)
    except Exception:
        try:
            h = [x for x in trainer.state.log_history if "step" in x]
            return int(h[-1]["step"])
        except Exception:
            return None


print(f"publishing {REPO}\n  branch {BRANCH}\n  wip    {WIP}")

# ---------------------------------------------------------------- 0. the card
# Fix the source field BEFORE any save, so every README generated below is born
# valid; hub_safe_card is the belt for anything that was written before this cell.
if os.path.isdir(str(getattr(model.config, "_name_or_path", ""))):
    model.config._name_or_path = UPSTREAM_ID
print(f"base_model for generated cards: {model.config._name_or_path}")

# ------------------------------------------------- 1. final WIP adapter push
# First, and on its own: it is small (~200 MB), it is the artifact that makes a
# reset recoverable to a specific step, and everything after it can fail without
# costing anything.
_s = step_now()
tag = f"ckpt-s{(_s or 0) // 100 * 100:05d}" + ("-final" if _s and _s % 100 else "")
d = f"/tmp/{tag}"
try:
    shutil.rmtree(d, ignore_errors = True)
    model.save_pretrained(d)
    tokenizer.save_pretrained(d)
    _card = hub_safe_card(d)
    print("wip card:", _card)
    upload_folder(folder_path = d, repo_id = REPO, repo_type = "model",
                  revision = WIP, commit_message = tag)
    print(f"PUSHED {tag} @ step {_s} -> {WIP}")
except Exception as e:
    print(f"wip push FAILED ({type(e).__name__}: {e})")
finally:
    shutil.rmtree(d, ignore_errors = True)

# ------------------------------------------------- 2. merged weights on BRANCH
print(ensure_branch(REPO, BRANCH, source = WIP))
merged_dir = "/tmp/final-merged"
try:
    if not os.path.exists(os.path.join(merged_dir, "model.safetensors")):
        model.generation_config.max_length = None      # do not bake this into the GGUF
        model.generation_config.max_new_tokens = None
        shutil.rmtree(merged_dir, ignore_errors = True)
        model.save_pretrained_merged(merged_dir, tokenizer = tokenizer,
                                     save_method = "merged_16bit")
        print("merged 16-bit written to", merged_dir)
    print("merged card:", hub_safe_card(merged_dir))
    upload_folder(folder_path = merged_dir, repo_id = REPO, repo_type = "model",
                  revision = BRANCH, commit_message = f"merged_16bit @ step {_s}")
    print(f"PUSHED merged_16bit -> {BRANCH}")
except Exception as e:
    print(f"merged push FAILED ({type(e).__name__}: {e}) - the GGUF below is still worth doing")

# ------------------------------------------------------------ 3. GGUF on BRANCH
# unsloth writes its f16 intermediate to the CWD and then moves it into the export
# dir; if those are on different filesystems that move is a multi-GB copy. /tmp has
# the space. The GGUF directory may already exist if the notebook's own publish cell
# got that far before it died on the card - reuse it rather than re-export.
gguf_dir = "/tmp/final-gguf"
os.makedirs(gguf_dir, exist_ok = True)
_free = shutil.disk_usage("/tmp").free / 2**30
print(f"/tmp free: {_free:.0f} GB")
try:
    if not any(f.endswith(".gguf") for f in os.listdir(gguf_dir)):
        _cwd = os.getcwd()
        os.chdir("/tmp")
        try:
            paths = model.save_pretrained_gguf(gguf_dir, tokenizer,
                                               quantization_method = "q4_k_m")
        finally:
            os.chdir(_cwd)
        print("gguf exported:", paths)
    print("gguf card:", hub_safe_card(gguf_dir))
    upload_folder(folder_path = gguf_dir, repo_id = REPO, repo_type = "model",
                  revision = BRANCH, allow_patterns = ["*.gguf", "*.json", "*.md"],
                  commit_message = f"gguf q4_k_m @ step {_s}")
    print(f"PUSHED gguf q4_k_m -> {BRANCH}")
except Exception as e:
    print(f"gguf push FAILED ({type(e).__name__}: {e})")

print("\ndone ->", f"https://huggingface.co/{REPO}/tree/{BRANCH}")
print("wip   ->", f"https://huggingface.co/{REPO}/tree/{WIP}")