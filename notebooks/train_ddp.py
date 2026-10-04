#!/usr/bin/env python3
"""Hindi CPT + SFT for LFM2.5-1.2B under Distributed Data Parallel.

Launched by the notebook - one process per GPU - as:

    python -m torch.distributed.run --standalone --nproc_per_node=2 train_ddp.py --stage cpt
    python -m torch.distributed.run --standalone --nproc_per_node=2 train_ddp.py --stage sft

Why DDP and not the old single process. one card trained and the second T4 sat
idle. The old notebook's comment blamed a hard block in Unsloth -
`tokenizer_utils.py:1813` "does not support multi GPU". That block is dead code
in the installed release: `patch_sft_trainer_tokenizer()` is defined but has no
call site in unsloth-2026.9.14, so it is never installed on `SFTTrainer.train`.
The documented path (unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) is
DDP via `torchrun`, and Unsloth's own device planner pins each rank to its card
as soon as `WORLD_SIZE > 1`. Two T4s then train on distinct shards and all-reduce
the LoRA gradients once per optimizer step, doubling rows/step at the same
effective batch.

`device_map` is deliberately NOT passed: unsloth's planner reads LOCAL_RANK and
resolves `{"": "cuda:<rank>"}` per process, which is the one placement that keeps
the two ranks off each other's card. Only rank 0 writes to the Hub - the
wall-clock checkpoint callback, the merged CPT push, the final merge -> GGUF -
so two processes never race on one revision.

Every number comes from /kaggle/working/ddp_config.json, written by the notebook,
so the notebook stays the single place a run is configured.
"""
import argparse
import gc
import json
import os
import time
import urllib.request

CFG_PATH = os.environ.get("DDP_CONFIG", "/kaggle/working/ddp_config.json")
CFG = json.load(open(CFG_PATH, encoding="utf-8"))

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN = LOCAL_RANK == 0

# unsloth first, before any transformers import in this process, or it warns and
# skips its patches. The model is placed by unsloth's distributed planner from
# LOCAL_RANK/WORLD_SIZE - no explicit device_map, which would fight the planner.
import torch  # noqa: E402  (after reading CFG; torch does not touch transformers)

torch.cuda.set_device(LOCAL_RANK)

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402
from transformers import AutoTokenizer, TrainerCallback  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402
from peft import PeftModelForCausalLM  # noqa: E402

DTYPE = torch.float16 if CFG["DTYPE"] == "float16" else torch.bfloat16


def log(msg):
    print(f"[rank {LOCAL_RANK}/{WORLD_SIZE}] {msg}", flush=True)


class TimeCheckpoint(TrainerCallback):
    """Push the adapter + tokenizer to the Hub every `every_s` wall-clock seconds.

    Rank 0 only: the callback runs on every process, and two writers to one repo
    revision race. `state.is_world_process_zero` is the DDP-safe form of
    `local_rank == 0`.
    """

    def __init__(self, model, repo, every_s, tokenizer=None, revision=None):
        self.model, self.repo, self.every = model, repo, every_s
        self.rev, self.tok, self.last = revision, tokenizer, time.time()

    def on_step_end(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return control
        if time.time() - self.last >= self.every:
            tag = f"ckpt-s{state.global_step // 100 * 100:05d}"
            self.model.push_to_hub(self.repo, save_name=tag, revision=self.rev, private=True)
            if self.tok is not None:
                self.tok.push_to_hub(self.repo, save_name=tag, revision=self.rev, private=True)
            log(f"pushed {tag} @ step {state.global_step}")
            self.last = time.time()
        return control


def merge_if_adapter(model):
    """A repo publishing merged weights AND an adapter makes from_pretrained
    attach the adapter; get_peft_model then refuses with "You already added LoRA
    adapters". Merge it in first instead of applying it twice."""
    if isinstance(model, PeftModelForCausalLM):
        log("base carries an adapter - merging it before attaching a new one")
        model = model.merge_and_unload()
    return model


def fetch(url, dest):
    tok = os.environ.get("HF_TOKEN", "").strip()
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=600) as r:
        body = r.read()
    with open(dest, "wb") as f:
        f.write(body)
    return dest


def push_license(repo, revision):
    """Liquid Open License v1.0 has 16 redistribution clauses; the text travels
    with the weights, so it goes onto the branch that carries them."""
    from huggingface_hub import upload_file
    url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/resolve/main/LICENSE"
    with urllib.request.urlopen(url, timeout=180) as r:
        lic = r.read()
    with open("/tmp/CPT_LICENSE", "wb") as f:
        f.write(lic)
    upload_file(path_or_fileobj="/tmp/CPT_LICENSE", path_in_repo="LICENSE",
                repo_id=repo, revision=revision)
    log(f"LICENSE pushed to {repo}@{revision} ({len(lic):,} bytes)")


def _target_missing(model, targets):
    names = [n for n, _ in model.named_modules()]
    return [m for m in targets if not any(n == m or n.endswith("." + m) for n in names)]


# --------------------------------------------------------------------------- CPT
def run_cpt():
    # CCE (cut cross entropy) is unsupported for continued pre-training; must be
    # set before the trainer is built.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    cpt_tok = AutoTokenizer.from_pretrained(CFG["TOK_DIR"])
    assert cpt_tok.eos_token, "no eos_token; CPT would never learn to stop"
    eos = cpt_tok.eos_token

    dest = fetch(f'https://huggingface.co/datasets/{CFG["DATA_REPO"]}/resolve/main/{CFG["CPT_FILE"]}',
                 "/tmp/cpt_sample.jsonl")
    docs = [json.loads(l)["text"] for l in open(dest, encoding="utf-8") if l.strip()]
    cpt_ds = Dataset.from_dict({"text": [d + eos for d in docs]})
    log(f"CPT: {len(cpt_ds):,} documents")

    cpt_model, _ = FastLanguageModel.from_pretrained(
        model_name = CFG["BASE_REPO"], tokenizer_name = CFG["TOK_DIR"],
        max_seq_length = CFG["MAX_SEQ_LENGTH"], dtype = DTYPE,
        load_in_4bit = False, max_lora_rank = CFG["CPT_R"],
    )
    cpt_model = merge_if_adapter(cpt_model)

    # embed_tokens/lm_head are the mechanism for the new script (vendor: "add
    # embed_tokens and lm_head to allow the model to learn out of distribution
    # data"). LFM ties them, so Unsloth moves them to modules_to_save.
    cpt_targets = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3",
                   "embed_tokens", "lm_head"]
    missing = _target_missing(cpt_model, cpt_targets)
    if missing:
        raise RuntimeError(f"CPT target modules not present: {missing}")

    cpt_model = FastLanguageModel.get_peft_model(
        cpt_model, r = CFG["CPT_R"], lora_alpha = CFG["CPT_ALPHA"], lora_dropout = 0,
        bias = "none", target_modules = cpt_targets,
        use_gradient_checkpointing = "unsloth", random_state = 3407,
        use_rslora = True,   # True for CPT, False for SFT - the vendor differs deliberately
    )
    n = sum(p.numel() for p in cpt_model.parameters() if p.requires_grad)
    if n == 0:
        raise RuntimeError("CPT LoRA attached to zero parameters - check cpt_targets")
    log(f"CPT trainable {n/1e6:.1f}M")

    cpt_trainer = UnslothTrainer(
        model = cpt_model, tokenizer = cpt_tok,
        train_dataset = cpt_ds, eval_dataset = None,
        dataset_text_field = "text", max_seq_length = CFG["CPT_SEQ"],
        dataset_num_proc = 4,
        args = UnslothTrainingArguments(
            per_device_train_batch_size = CFG["CPT_BS"],
            gradient_accumulation_steps = CFG["CPT_ACC"],
            max_steps = CFG["CPT_STEPS"], warmup_steps = 10,
            learning_rate = CFG["CPT_LR"],
            embedding_learning_rate = CFG["CPT_EMB_LR"],
            logging_steps = max(5, CFG["CPT_STEPS"] // 60),
            optim = "adamw_8bit", weight_decay = 0.001, lr_scheduler_type = "linear",
            seed = 3407, output_dir = "cpt-out", save_strategy = "no", report_to = [],
        ),
    )
    cpt_trainer.add_callback(TimeCheckpoint(cpt_model, CFG["REPO"], CFG["CHECKPOINT_EVERY_S"],
                                            tokenizer = cpt_tok, revision = CFG["WIP"]))
    if IS_MAIN:
        log(f"CPT rows/step = {CFG['CPT_BS']} x {CFG['CPT_ACC']} x {WORLD_SIZE} GPU "
            f"= {CFG['CPT_BS'] * CFG['CPT_ACC'] * WORLD_SIZE}")
    cpt_stats = cpt_trainer.train()
    log(f"CPT metrics: {cpt_stats.metrics}")

    if IS_MAIN:
        cpt_model.generation_config.max_length = None
        cpt_model.generation_config.max_new_tokens = None
        # Publish the merged fp16 model AND the adapter, then hand the merged dir
        # to the SFT stage. This is the cpt-hindi branch the chain is built on.
        cpt_model.save_pretrained_merged(CFG["SFT_BASE"], tokenizer = cpt_tok,
                                         save_method = "merged_16bit")
        cpt_model.push_to_hub(CFG["REPO"], tokenizer = cpt_tok, revision = "cpt-hindi")
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id = CFG["REPO"], revision = "cpt-hindi",
                              folder_path = CFG["SFT_BASE"],
                              ignore_patterns = ["*.pth", "cpt-out/*"])
        push_license(CFG["REPO"], "cpt-hindi")
        log(f"CPT done -> https://huggingface.co/{CFG['REPO']}/tree/cpt-hindi")

    # Let rank 0 finish writing the merged dir before the process group is torn
    # down; the SFT stage then loads it.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# --------------------------------------------------------------------------- SFT
def run_sft():
    tok = AutoTokenizer.from_pretrained(CFG["TOK_DIR"])
    split = load_from_disk(CFG["SFT_SPLIT"])
    train_ds, eval_ds = split["train"], split["test"]
    log(f"SFT: train {len(train_ds):,} eval {len(eval_ds):,} rows")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = CFG["SFT_BASE"], tokenizer_name = CFG["TOK_DIR"],
        max_seq_length = CFG["MAX_SEQ_LENGTH"], dtype = DTYPE,
        load_in_4bit = False, max_lora_rank = 32,
    )
    model = merge_if_adapter(model)

    # LFM's MLP is w1/w2/w3 and its conv layers own in_proj/out_proj - not the
    # usual LLaMA names. A wrong list attaches nothing and still trains "fine".
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
    missing = _target_missing(model, target_modules)
    if missing:
        raise RuntimeError(f"Target modules not present in this model: {missing}")

    model = FastLanguageModel.get_peft_model(
        model, r = 16, lora_alpha = 16, lora_dropout = 0, bias = "none",
        target_modules = target_modules, use_gradient_checkpointing = "unsloth",
        random_state = 3407, use_rslora = False,
    )
    # SFT must not overwrite the embeddings CPT just fitted on 24,000 documents.
    et = [n for n, p in model.named_parameters() if "embed_tokens" in n and p.requires_grad]
    if et:
        raise RuntimeError(f"SFT is training {et} - it must leave the CPT embeddings alone.")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train == 0:
        raise RuntimeError("LoRA attached to zero parameters - check target_modules")
    log(f"SFT trainable {n_train/1e6:.1f}M")

    trainer = UnslothTrainer(
        model = model, tokenizer = tokenizer,
        train_dataset = train_ds, eval_dataset = eval_ds,
        dataset_text_field = "text", max_seq_length = CFG["MAX_SEQ_LENGTH"],
        dataset_num_proc = 2, packing = False,
        args = UnslothTrainingArguments(
            per_device_train_batch_size = CFG["SFT_BS"],
            gradient_accumulation_steps = CFG["SFT_ACC"],
            max_steps = CFG["MAX_STEPS"], warmup_steps = 5,
            learning_rate = CFG["LEARNING_RATE"], lr_scheduler_type = "linear",
            optim = "adamw_8bit", weight_decay = 0.01,
            logging_steps = max(5, CFG["MAX_STEPS"] // 60),
            eval_strategy = "steps", eval_steps = 150,
            seed = 3407, output_dir = "hf-out", save_strategy = "no",
            group_by_length = True,
            report_to = ["wandb"] if CFG["WANDB_ON"] else [],
            run_name = CFG["BRANCH"],
        ),
    )
    # Mask the user turn so loss is computed on the ASSISTANT RESPONSE ONLY. Without
    # it the model is trained to emit the question as well as the answer.
    trainer = train_on_responses_only(
        trainer, instruction_part = "<|im_start|>user\n", response_part = "<|im_start|>assistant\n")
    _lab = trainer.train_dataset[0]["labels"]
    _masked = sum(1 for x in _lab if x == -100)
    if _masked == 0:
        raise RuntimeError("train_on_responses_only did not mask anything")
    log(f"label masking: {_masked}/{len(_lab)} tokens are -100")
    trainer.add_callback(TimeCheckpoint(model, CFG["REPO"], CFG["CHECKPOINT_EVERY_S"],
                                        tokenizer = tokenizer, revision = CFG["WIP"]))

    if IS_MAIN:
        log(f"SFT rows/step = {CFG['SFT_BS']} x {CFG['SFT_ACC']} x {WORLD_SIZE} GPU "
            f"= {CFG['SFT_BS'] * CFG['SFT_ACC'] * WORLD_SIZE}")

    torch.cuda.reset_peak_memory_stats()
    stats = trainer.train()
    log(f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB "
        f"of {torch.cuda.get_device_properties(LOCAL_RANK).total_memory/2**30:.1f} GB")

    if IS_MAIN:
        print(str(stats.metrics), flush=True)
        print_eval(trainer)
        merge_gguf_and_push(model, tokenizer)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def print_eval(trainer):
    """The eval-loss minimum is what decides the next run's step count, so print
    the curve and name the minimum instead of leaving it in a table."""
    hist = [(h.get("step"), h.get("loss"), h.get("eval_loss")) for h in trainer.state.log_history]
    ev = [(s, e) for s, l, e in hist if e is not None and s is not None]
    if not ev:
        return
    best = min(ev, key=lambda x: x[1])
    print(f"eval loss: {[(s, round(l, 4)) for s, l in ev]}", flush=True)
    print(f"  minimum at step {best[0]} = {best[1]:.4f}", flush=True)
    if ev[-1][1] > best[1]:
        print(f"  next run should stop near step {best[0]}, not {CFG['MAX_STEPS']}", flush=True)


def merge_gguf_and_push(model, tokenizer):
    import shutil
    from huggingface_hub import create_branch, model_info
    from huggingface_hub.utils import RevisionNotFoundError
    try:
        model_info(CFG["REPO"], revision=CFG["BRANCH"])
    except RevisionNotFoundError:
        log(f"branch '{CFG['BRANCH']}' not found - creating it")
        create_branch(CFG["REPO"], branch=CFG["BRANCH"], exist_ok=True)

    # Unsloth writes the f16 conversion to the cwd then moves it; on different
    # filesystems that is a 4.2 GB copy of a dir with little space, so chdir to
    # /tmp where the room is.
    shutil.rmtree("/kaggle/working/teacher", ignore_errors=True)
    os.chdir("/tmp")
    free = shutil.disk_usage("/tmp").free / 2**30
    print(f"cwd {os.getcwd()}  {free:.0f} GB free", flush=True)
    if free < 10:
        raise RuntimeError(f"only {free:.0f} GB on /tmp - not enough for the f16 pass")

    model.generation_config.max_length = None
    model.generation_config.max_new_tokens = None
    p = model.save_pretrained_gguf("hf-out", tokenizer, quantization_method = "q4_k_m")
    print("gguf at:", p, flush=True)
    model.push_to_hub_gguf(CFG["REPO"], tokenizer, quantization_method = "q4_k_m",
                           revision = CFG["BRANCH"])
    model.push_to_hub(CFG["REPO"], tokenizer, revision = CFG["BRANCH"])
    print(f"done -> https://huggingface.co/{CFG['REPO']}/tree/{CFG['BRANCH']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cpt", "sft"], required=True)
    a = ap.parse_args()
    log(f"stage={a.stage} world_size={WORLD_SIZE}")
    if a.stage == "cpt":
        run_cpt()
    else:
        run_sft()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
