#!/usr/bin/env python3
"""
Hindi CPT + SFT DDP training script for LFM2.5-1.2B
Compatible with both unsloth and standard HuggingFace

Usage (in Kaggle notebook):
  !torchrun --nproc_per_node=2 notebooks/train_ddp_v8.py
"""
import os
import json
import time
import torch
import gc

# DDP setup
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN = LOCAL_RANK == 0
os.environ["RANK"] = str(LOCAL_RANK)
os.environ["WORLD_SIZE"] = str(WORLD_SIZE)

def log(msg):
    print(f"[rank {LOCAL_RANK}/{WORLD_SIZE}] {msg}", flush=True)

# Try unsloth first, fall back to HF
UNSLOTH_AVAILABLE = False
try:
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
    from unsloth.chat_templates import train_on_responses_only
    UNSLOTH_AVAILABLE = True
    log("unsloth imported - DDP mode ready")
except Exception as e:
    log(f"unsloth import failed: {e}")
    log("falling back to standard HuggingFace")
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModelForCausalLM
    
    class FastLanguageModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tok = kwargs.pop("tokenizer", None)
            model = AutoModelForCausalLM.from_pretrained(*args, **kwargs)
            return model, tok
        
        @staticmethod
        def get_peft_model(model, **kwargs):
            r = kwargs.get("r", kwargs.get("max_lora_rank", 64))
            alpha = kwargs.get("lora_alpha", 16)
            do = kwargs.get("lora_dropout", 0)
            tm = kwargs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            use_gc = kwargs.get("use_gradient_checkpointing", "unsloth")
            if use_gc == "unsloth":
                use_gc = True
            else:
                use_gc = str(use_gc).lower() in ("true", "1", "yes")
            if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
                model = prepare_model_for_kbit_training(model)
            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=do, target_modules=tm,
                           bias=kwargs.get("bias", "none"), use_gradient_tokenizers=use_gc,
                           task_type="CAUSAL_LM")
            return get_peft_model(model, cfg)
        
        @staticmethod
        def save_pretrained_merged(model, out_dir, **kwargs):
            save_m = kwargs.get("save_method", "merged_16bit")
            if save_m == "merged_16bit":
                model = model.merge_and_unload()
            model.save_pretrained(out_dir, safe_serialization=True, torch_dtype=torch.float16)
            return model

    class UnslothTrainer(Trainer):
        pass
    
    class UnslothTrainingArguments(TrainingArguments):
        pass
    
    def train_on_responses_only(*args, **kwargs):
        log("train_on_responses_only not available in fallback mode")
        return args[0] if args else None

# Main training logic
torch.cuda.set_device(LOCAL_RANK)
DTYPE = torch.float16

def fetch(url, dest):
    tok = os.environ.get("HF_TOKEN", "").strip()
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=600) as r:
        with open(dest, "wb") as f:
            f.write(r.read())
    return dest

def run_sft():
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained("zaakirio/LFM2.5-1.2B-Instruct-Uncensored")
    
    # Load base model CPT-adapted
    model, tok = FastLanguageModel.from_pretrained(
        model_name="zaakirio/LFM2.5-1.2B-Instruct-Uncensored",
        tokenizer_name="zaakirio/LFM2.5-1.2B-Instruct-Uncensored",
        max_seq_length=4096, dtype=DTYPE, load_in_4bit=False, max_lora_rank=32
    )
    
    # Merge if adapter present
    if isinstance(model, PeftModelForCausalLM):
        log("base carries an adapter - merging before attaching new LoRA")
        model = model.merge_and_unload()
    
    # Apply LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
    model = FastLanguageModel.get_peft_model(
        model, r=64, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=target_modules, use_gradient_checkpointing="unsloth",
        random_state=3407
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable {n_params/1e6:.2f}M params")
    
    # Load dataset
    import glob
    candidates = glob.glob("/kaggle/working/train_v8.jsonl")
    if not candidates:
        raise RuntimeError("train_v8.jsonl not found - run data prep first")
    
    from datasets import load_dataset
    ds = load_dataset("json", data_files={"train": candidates[0]}, split="train")
    log(f"dataset: {len(ds)} rows")
    
    # Training args
    args = UnslothTrainingArguments(
        output_dir="/kaggle/working/sft_out",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=4025 // WORLD_SIZE,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    
    # Checkpoint callback
    from transformers import TrainerCallback
    class CCP:
        def __init__(self, mod, every=900):
            self.mod, self.every, self.last = mod, every, time.time()
        def on_step_end(self, args, state, control, **kw):
            if state.is_world_process_zero and (time.time() - self.last >= self.every):
                tag = f"ckpt-s{state.global_step//100*100:05d}"
                self.mod.push_to_hub("kumarsujitroy/LFM2.5-1.2B-Instruct-HI-Uncensored",
                    save_name=tag, revision="wip/s4-hinglish-v8", private=True)
                log(f"pushed {tag}")
                self.last = time.time()
            return control
    
    trainer = UnslothTrainer(model=model, args=args, train_dataset=ds)
    trainer.add_callback(CCP(model))
    
    log("starting training...")
    trainer.train()
    
    if IS_MAIN:
        log("training complete")
        model.push_to_hub("kumarsujitroy/LFM2.5-1.2B-Instruct-HI-Uncensored",
            private=True, revision="s4-hinglish-v8")
        tok.push_to_hub("kumarsujitroy/LFM2.5-1.2B-Instruct-HI-Uncensored",
            private=True, revision="s4-hinglish-v8")

if __name__ == "__main__":
    run_sft()