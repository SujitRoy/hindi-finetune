# LFM2.5-1.2B-Instruct-Uncensored → Hindi + Hinglish

Fine-tune Liquid AI's `LFM2.5-1.2B-Instruct-Uncensored` to speak **Devanagari Hindi and
romanized Hinglish**, on Kaggle's free 2× T4, in sessions that die every 30–40 minutes.

Everything here is stdlib-only Python for data prep. The GPU work is a single notebook.

---

## Table of contents

1. [Why this model](#1-why-this-model)
2. [The tokenizer finding — the constraint that shapes everything](#2-the-tokenizer-finding--the-constraint-that-shapes-everything)
3. [Repo layout](#3-repo-layout)
4. [The data pipeline, A to Z](#4-the-data-pipeline-a-to-z)
5. [Kaggle setup from scratch](#5-kaggle-setup-from-scratch)
6. [The notebook, cell by cell](#6-the-notebook-cell-by-cell)
7. [Chained sessions and crash-proof checkpointing](#7-chained-sessions-and-crash-proof-checkpointing)
8. [Bringing the model back to the server](#8-bringing-the-model-back-to-the-server)
9. [Acceptance gates — do not skip these](#9-acceptance-gates--do-not-skip-these)
10. [Troubleshooting](#10-troubleshooting)
11. [File inventory](#11-file-inventory)

---

## 1. Why this model

The target is a production ARM64 server, so the viable envelope was measured, not
guessed:

| model | size | tok/s | RAM | accuracy | verdict |
|---|---|---|---|---|---|
| **LFM2.5-1.2B-Instruct-Uncensored-Q4_K_M** | 0.73 GB | 14.7 | 1.4 GB | 5/8 | **chosen** |
| LFM2.5-1.2B-Uncensored Q6_K | 0.92 GB | 9.6 | 1.8 GB | 4/8 | quant is *worse* |
| LFM2.5-2.6B-UNCENSORED-ABLITERATED-PHILLY | 1.6 GB | 8.1 | 2.9 GB | 6/8 | too slow |
| LFM2.5-2.6B-Uncensored | 1.6 GB | 8.4 | 2.9 GB | 7/8 | too slow |

Only the 1.2B is genuinely usable as a live server. Inside 1–1.5B the prompt/quant/
history levers are exhausted — 4 system-prompt variants gave 5/5 instruction-following,
and the remaining accuracy gap is parameter count, nothing else. Fine-tuning is the only
remaining lever, which is what this repo is for.

**The base model is the abliterated (uncensored) one, deliberately.** Fine-tuning
regular instructions would slowly re-teach refusal. That regression is the single most
likely failure of this project — hence gate 9.3.

### The GGUF you have is not what gets trained

GGUF is a quantised inference format. It cannot be trained. So:

```
zaakirio/LFM2.5-1.2B-Instruct-Uncensored   (safetensors, 2.4 GB)  <- trained here
        |  train with LoRA, merge
        v
merged safetensors
        |  save_pretrained_gguf("q4_k_m")   (llama.cpp converter, in-notebook)
        v
LFM2.5-1.2B-…Q4_K_M.gguf  (0.73 GB)       <- what the server loads
```

The notebook does this conversion in-notebook. That step is where people normally get
stuck; it is cell 18 here, and it is the long pole for the 30-minute budget.

---

## 2. The tokenizer finding — the constraint that shapes everything

This is the single most important fact in the repo, and it was measured, not assumed.

```
नमस्ते, आप कैसे हैं?        20 chars -> 29 tokens   1.45 tok/char
   pieces: ['à¤', '¨', 'à¤', '®', 'à¤', '¸', 'à¥įà¤', '¤', 'à¥ĩ']   <- raw UTF-8 bytes
namaste, aap kaise hain?   24 chars -> 11 tokens   0.46 tok/char
```

The LFM tokenizer is a **byte-level BPE** (GPT-2 style: `Sequence[Split(GPT-2 regex),
ByteLevel]`, 64,402 tokens). It has **no dedicated Devanagari tokens** — Hindi falls
through to individual UTF-8 bytes. Consequences:

* Devanagari costs **3.2× more tokens** than the same text romanized. Not fatal, but it
  sets `MAX_SEQ_LENGTH` and how much Hindi fits in a session.
* `embed_tokens` has no Devanagari row to fine-tune; the model learns Hindi purely
  through the existing byte tokens. This is exactly how GPT-2 and Llama handle every
  non-Latin script — it works, it is just data-hungry.
* **This is why the dataset is Hinglish-heavy and Hindi is chrF-filtered**, not an
  arbitrary weighting choice.

### The failed tokenizer extension, and why

`extend_tokenizer.py` trains a small Devanagari BPE and splices the pieces into the
vocab. It runs, it round-trips, and **it does not help** — tokenization comes out
byte-identical to the base.

```python
bt = Tokenizer.from_str(json.dumps(base_json))   # correct pre-tokenizer
bt.model = models.BPE(unk_token=None)
bt.train_from_iterator(enc, ...)
```

Getting it to work needs merge-rank surgery across 64,140 merges, because the base's
existing higher-priority merges win first. Two bugs are documented in the file because
both are non-obvious and both produce a silent no-op:

1. **A vocab entry without a merge is unreachable.** BPE can only emit tokens it has a
   merge rule for. Adding vocabulary alone does nothing.
2. **The pre-tokenizer must match exactly.** Training on plain `ByteLevel` learns merges
   that span word boundaries; at inference the GPT-2 regex splits first, so those merges
   can never fire. The first version of this script made that mistake.

The file is kept because the diagnosis is valuable and the technique is the right one if
you ever need a large-vocab extension. **It is not on the training path.** Byte-level
works; it just costs tokens.

---

## 3. Repo layout

```
hindi-finetune/
├── README.md                     this file
├── prepare_data.py               builds every training file (stdlib + requests)
├── verify_format.py              validates a jsonl against the REAL tokenizer
├── extend_tokenizer.py           Devanagari vocab extension (experimental, off-path)
├── hinglish-sft-lfm25-1.2b.ipynb the training notebook
├── hi_corpus.txt                 chrF-filtered Devanagari, for the tokenizer experiment
│
├── train_s1_hinglish.jsonl       5,500 rows   session 1
├── train_s2_hindi.jsonl          6,000 rows   session 2  (Devanagari)
├── train_s3_english.jsonl        5,000 rows   session 3
│
├── train_kaggle30.jsonl          one-session bilingual alternative
├── train_smoke.jsonl             15 rows, pipeline test in ~5 s
│
├── ds_arena_hinglish.jsonl       per-source, so re-weighting needs no re-download
├── ds_casual_hinglish.jsonl
├── ds_english_dolly.jsonl
├── ds_hindi_devanagari.jsonl
│
├── tokenizer-hi/                 experimental extended tokenizer (off-path)
└── cookbook/                     Liquid's official finetuning notebook + LoRA script
```

`ds_*.jsonl` files exist so you can re-mix the corpus in seconds instead of re-downloading
from the Hub. Every build is ~5 s (Hinglish) or ~5 min (Hindi, the rows API is slow).

---

## 4. The data pipeline, A to Z

```bash
cd ~/hindi-finetune
python3.13 prepare_data.py --tier s1_hinglish     # session 1
python3.13 prepare_data.py --tier s2_hindi        # session 2
python3.13 prepare_data.py --tier s3_english      # session 3
python3.13 verify_format.py train_s1_hinglish.jsonl   # always
```

### 4.1 Sources

| source | dataset | licence | why |
|---|---|---|---|
| `arena` | `one-thing/chatbot_arena_conversations_hinglish` | apache-2.0 | real human prompts from chatbot arena, with substance in the replies. The backbone. |
| `casual` | `Abhishekcr448/Hinglish-Everyday-Conversations-1M` | mit | natural romanized register, schwa-stripped. Small dose only — replies average 6.8 words. |
| `hindi` | `ai4bharat/indic-instruct-data-v0.1` (oasst1/hi, dolly/hi) | — | Devanagari. Chosen because it ships **per-row `quality_metrics`** (chrF/sacreBLEU), which makes it filterable. |
| `english` | `databricks/databricks-dolly-15k` | cc-by-sa-3.0 | anti-forgetting anchor |

The Hindi dataset's configs:

| config | rows | note |
|---|---|---|
| `oasst1` | 20,128 | real OpenAssistant Hindi conversations, chat-native. Best for chat ability. **8,799 pass chrF≥70.** |
| `dolly` | 15,011 | Hindi twin of the English dolly we already use, with backtranslation. **13,657 pass chrF≥70.** |
| `flan_v2` | 67,463 | **skipped** — badly translated; sampled targets are things like `1)।` |
| `wikihow` | 6,055 | **skipped** — different parquet schema (`title/intro/steps`), no `quality_metrics`, so it cannot be chrF-filtered |

> Parquet shards are pulled straight off the HF CDN, **not** the datasets-server rows
> API. The rows API hard-**429s** after a few hundred pages and each retry costs more
> than downloading the few-MB shard.

> **Note on `ai4bharat/ai2_arc-hi`:** this is science **MCQ eval** data, not chat. It is
> useful to *evaluate* Hindi comprehension, not to train conversational ability. Correct
> shape, wrong purpose — hence not in the pipeline.

### 4.2 The `clean()` filter — four rejections

Each of these was added because a specific bad row reached the dataset and would have
poisoned training.

```python
DEV      = re.compile(r"[\u0900-\u097F]")   # any Devanagari codepoint
CODE_FENCE = re.compile(r"```")
REFUSAL  = re.compile(r"\b(i'?m sorry|as an ai\b|...)", re.I)
```

| rejection | why it matters |
|---|---|
| **Devanagari** in the Hinglish/English path | guarantees romanized and Devanagari data never get mixed by accident. `allow_devanagari=True` for the Hindi source only. |
| **Code fences** | the arena CSVs mangle code blocks; the model would learn broken markdown |
| **Refusals** ⚠️ | the translated sources contain the *original* model refusing. Training on those teaches your uncensored model to refuse — the exact opposite of the goal. Caught by `verify_format.py` finding a translated arena reply: *"I am not able to translate… That word is not appropriate."* |
| **Empty / runaway** | < 2 words, or > 120 words (dolly's `context` field makes RAG-style prompts) |

### 4.3 Tiers

`TIERS` in `prepare_data.py` is `(arena, casual, hindi, english)`:

| tier | arena | casual | hindi | english | rows | use |
|---|---|---|---|---|---|---|
| `smoke` | 5 | 5 | 5 | 5 | 15 | pipeline test, 5 s |
| `kaggle30` | 4000 | 1500 | 3500 | 1000 | ~6.3k | one-session bilingual alternative |
| `s1_hinglish` | 4000 | 1500 | 0 | 0 | 5500 | **session 1** |
| `s2_hindi` | 0 | 0 | 6000 | 0 | 6000 | **session 2** |
| `s3_english` | 0 | 0 | 0 | 5000 | 5000 | **session 3** |
| `full` | 11000 | 6000 | 12000 | 5000 | ~24k | one long run (Colab) |

### 4.4 Why English is a whole session

Pure-Hinglish training would shift the abliterated weights and regress English *and*
uncensored compliance. Session 3 exists specifically to pull the model back toward
English. Drop it and you will find out the hard way.

### 4.5 Verification — never skip this

```bash
python3.13 verify_format.py train_s1_hinglish.jsonl
```

It loads the **real tokenizer**, applies the real chat template, and reports token
medians, p99, truncation rate, and the `train_on_responses_only` markers. **All three
session files: `FORMAT OK`, 0.00% truncation** at `MAX_SEQ_LENGTH=1024`.

| file | rows | median | p90 | max | truncates |
|---|---|---|---|---|---|
| `train_s1_hinglish.jsonl` | 5,500 | 41 | 112 | 610 | 0.00% |
| `train_s2_hindi.jsonl` | 6,000 | 372 | 685 | 811 | 0.00% |
| `train_s3_english.jsonl` | 4,087 | 95 | 218 | 761 | 0.00% |

### ⚠️ The Hindi truncation trap (this one would have cost a whole session)

Hindi rows are **9× the token median** of Hinglish, and not only because of the 1.45
tok/char penalty — the dolly `context` field and verbose oasst1 answers make the rows
genuinely long. The first unfiltered build measured:

```
6000 rows | median 1110, p90 2529, max 4632
truncation at max_seq_length=1024: 52.67%      <-- over half the dataset
```

**A truncated target teaches the model to never finish its reply.** The fix is a
character cap in the Hindi path only:

```python
HINDI_MAX_CHARS = 620     # 620 * 1.45 + ~15 template tokens ~= 915
```

which brings the median to 372 and truncation to zero. This is why `clean()` takes
`max_chars` rather than a word count — a word cap is meaningless when words tokenize
into multi-byte soup.

### Format per row:

```json
{"instruction": "...", "response": "...", "src": "arena"}
```

which becomes

```
<|startoftext|><|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n…<|im_end|>\n
```

⚠️ `apply_chat_template` **adds `<|startoftext|>` itself** and terminates with
`<|im_end|>`. Do not append `EOS_TOKEN` (most tutorials do, and it double-terminates)
and do not strip the BOS twice.

---

## 5. Kaggle setup from scratch

### 5.1 Hugging Face account + token (do this first)

1. Sign up at <https://huggingface.co>
2. Avatar -> **Settings -> Access Tokens -> Create new token** -> grant **`write`**

`push_to_hub` is the *only* thing that survives a Kaggle reset. Without a token every
checkpoint push fails and the run is lost.

Three repos are created automatically on first push:

```
https://huggingface.co/<you>/lfm25-1.2b-bilingual-s1
https://huggingface.co/<you>/lfm25-1.2b-bilingual-s2
https://huggingface.co/<you>/lfm25-1.2b-bilingual-s3
```

#### The notebook is public, so there is no paste slot for a token

Anything typed into a public notebook is published. The token has to arrive as an
**environment variable**, which Kaggle injects at runtime and never writes into the
notebook JSON.

**Setup, once:**

| step | where | what |
|---|---|---|
| 1 | right sidebar -> **Add-ons** tab | open the Secrets section |
| 2 | **+ New Secret** | Label `HF_TOKEN`, Value your `hf_...` token |
| 3 | same dialog | **tick "Notebook Access"** |
| 4 | **Session** tab | **restart** - Kaggle injects secrets only at session start |

If you cannot see an `Add-ons` tab, the same panel is under the **top menu bar ->
Add-ons -> Secrets**. Kaggle moves this between UI revisions.

Verify before a long run:

```python
import os
print("set:", bool(os.environ.get("HF_TOKEN")), "| len:", len(os.environ.get("HF_TOKEN", "")))
```

`True` and ~37. If it prints `False`, the label is misspelled (it is case-sensitive) or
Notebook Access was left unticked.

### 5.2 Upload the training data as a Kaggle Dataset

The notebook reads from `/kaggle/input/...`, not from your laptop.

1. Kaggle → **Datasets** → **+ New Dataset** → upload
2. Upload `train_s1_hinglish.jsonl` (1.1 MB)
3. Name it `hinglish-sft` — the notebook's first candidate path expects that
4. Repeat for the other sessions as you get to them

### 5.3 The notebook

1. Kaggle → **Code** → **+ New Notebook** → **Add Input** → your `hinglish-sft` dataset
2. **Session → GPU T4 × 2** (right-hand panel — *not* the default CPU!)
3. **Settings → Internet**: **on**. Unsloth installs packages and downloads the base
   model at runtime; without it cell 5 fails.
4. Import `hinglish-sft-lfm25-1.2b.ipynb`, or paste the cells in order
5. **Edit one thing:** cell 1, `HF_USER = "YOUR_HF_USERNAME"` → your actual username
6. Set `SESSION = 1`. The token comes from the `HF_TOKEN` Kaggle secret — cell 4
   refuses to run without it, by design.

Then run all cells top to bottom. Run **one session per Kaggle session** — the notebook
is designed so a reset costs at most 5 minutes.

### 5.4 The one config change per session

| session | `SESSION` | loads from | data |
|---|---|---|---|
| 1 | `1` | the uncensored base | `train_s1_hinglish.jsonl` |
| 2 | `2` | `…-s1` | `train_s2_hindi.jsonl` |
| 3 | `3` | `…-s2` | `train_s3_english.jsonl` |

`OUT_REPO` and `BASE_REPO` are derived from `HF_USER` and `SESSION` — nothing else to edit.

---

## 6. The notebook, cell by cell

| # | cell | what it does | watch out for |
|---|---|---|---|
| 1 | **run config** | `SESSION`, `DATA_FILE`, `HF_USER`, `OUT_REPO`, `BASE_REPO`, `CHECKPOINT_EVERY_S`, `MAX_STEPS` | **the only cell you must edit** |
| 2–3 | HF login + where-it-lives | `login(token=…)` | required before any push |
| 4–5 | deps | `unsloth`, `transformers==4.57.3`, `trl==0.22.2` | needs internet |
| 6–7 | **`TimeCheckpoint`** | wall-clock checkpoint pusher | the crash-safety mechanism |
| 8–9 | load data | finds the jsonl under `/kaggle/input` | raises a clear error if not uploaded |
| 10 | lengths | `MAX_SEQ_LENGTH=1024`, **fp16** | T4 is sm_75 — **no bf16** |
| 11–12 | model + LoRA | r=32, LFM modules, **verified** | `embed_tokens`/`lm_head` deliberately excluded — see §2 |
| 13–14 | chat template | 2 turns, ends `<|im_end|>` | no manual EOS |
| 15 | trainer | eff. batch 8, lr 2e-4, linear | `save_strategy="no"` — the callback owns saving |
| 16 | train | the actual run | `MAX_STEPS` sized per session |
| 17–18 | merge → GGUF → push | `save_pretrained_gguf("q4_k_m")` | **the long pole for the 30-min budget** |
| 19–20 | **optional iMatrix GGUF** | `llama-imatrix` + `llama-quantize` | higher quality, same size; skip if short on time |
| 19 | crash recovery | how to resume from a pushed checkpoint | |

### LoRA configuration

```python
r = 32, lora_alpha = 32, use_rslora = True, lora_dropout = 0
target_modules = ["q_proj","k_proj","v_proj","out_proj","in_proj","w1","w2","w3"]
```

`in_proj` and `w1/w2/w3` are LFM's MLP; they are not the usual LLaMA names. `r=32` is
double the cookbook's SFT default of 16 — a language shift needs more adapter capacity
than a style tweak.

Learning rate is `2e-4` for session 1, and **lower for sessions 2–3** (the notebook's
comment flags this): chained merged fine-tuning drifts more than a single run, and
deviating from the base is the main risk.

### Cross-checked against the official Kaggle GRPO notebook

[`iamleonie/fine-tuning-lfm2-5-1-2b-instruct-with-grpo`](https://www.kaggle.com/code/iamleonie/fine-tuning-lfm2-5-1-2b-instruct-with-grpo)
finetunes this same base model on Kaggle. It is a **GRPO** run (RL with three reward
functions) on invoice-OCR→JSON extraction — a different technique and a different task,
so it is a *plumbing* reference, not a data one. What it confirms and what it does not:

| | reference notebook | this repo | why |
|---|---|---|---|
| LoRA rank | 32 | 32 | ✅ matches — good confirmation of the capacity choice |
| checkpoint safety | `save_pretrained` **once at the end** | `TimeCheckpoint` pushes to the Hub **every 5 min** | a single end-of-run save loses everything on a VM reset |
| method | GRPO + vLLM | SFT | GRPO needs vLLM for fast generation and far more VRAM; it does not fit 2×T4 free tier in 30 min |
| data | invoice OCR → JSON | Hindi + Hinglish chat | different task entirely |

Adopted from it: `UNSLOTH_VLLM_STANDBY=1` (−30% VRAM, matters on T4) and passing
`max_lora_rank` at load time. It pins `transformers 4.57.6` / `trl 0.24.0` where this
notebook uses `4.57.3` / `0.22.2` — minor drift, and if you hit an API error, matching
their versions is the first thing to try.

### Cross-checked against the Ava @ Dyagnosys LFM2.5 notebook

[`coachvitorcalvi/ava-lab-dyagnosys-finetune-v1`](https://www.kaggle.com/code/coachvitorcalvi/ava-lab-dyagnosys-finetune-v1)
is a production-shaped SFT run on the same 1.2B base (tool-calling for lab sales, fully
synthetic corpus). It is the better plumbing reference of the two. Adopted:

| from Ava | what | why it matters here |
|---|---|---|
| **target-module guard** | verify each name exists in `named_modules()` by suffix match *before* attaching LoRA, and hard-fail on zero trainable params | `named_modules()` returns `model.layers.0.self_attn.q_proj`, so a bare `"q_proj"` is **never** an exact member. A wrong name attaches nothing and the run trains happily on 0 parameters. Silent, total, and looks fine in the logs. |
| **iMatrix GGUF export** | f16 GGUF → `llama-imatrix` over a calibration corpus → `llama-quantize --imatrix … q4_k_m` | Same bit depth, same file size, less damage — plain `save_pretrained_gguf` picks outliers by tensor statistics alone. Directly relevant: the 1.2B scores 5/8, so we cannot afford to lose anything to quantization. Costs ~5–10 min; skip if near the 30-min mark. |
| **template-leakage smoke test** | assert train/eval `template_id` sets are disjoint | we use `train_test_split`; the invariant is still worth asserting |
| **OOD probes never in training** | hand-written, out-of-corpus | same idea as our `verify_format.py` + `/tmp/tune.py` |

It also **independently confirms the LoRA target module list** — verbatim
`["q_proj","k_proj","v_proj","out_proj","in_proj","w1","w2","w3"]`, and it uses effective
batch 8 like we do (though via `batch=1, accum=8`, which is slower than our `batch=8`).

One idea worth stealing later: it masks loss by **common-prefix diff** rather than by
assistant-token markers:

```python
split_at = common_prefix_len(prompt_ids, full_ids)
labels = [-100] * split_at + full_ids[split_at:]
```

Our `train_on_responses_only` path depends on the template's markers, which
`verify_format.py` already checks match verbatim — so both are safe today, but the
prefix-diff form is immune to a template change.

---

## 7. Chained sessions and crash-proof checkpointing

### Why chained

A single 24k-row run does not fit in 30–40 minutes. Splitting it into three sequential
runs, each starting from the previous run's **merged** output, uses the free tier
properly. The tradeoff is stated honestly: sequential merged fine-tuning drifts and is
harder to debug, so test after each session rather than only at the end.

### Why the callback, not `save_steps`

**A local save on Kaggle is not a backup.** `/kaggle/working` is destroyed when the VM
resets — every `save_steps` checkpoint dies with it. The only durable store is the Hub.

```python
class TimeCheckpoint(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if time.time() - self.last >= self.every:     # wall clock, not step count
            self.model.save_pretrained(f"hf-out/{tag}")
            self.model.push_to_hub(self.repo, save_name=tag)
```

`save_strategy` is set to `"no"` so the two mechanisms cannot fight. A LoRA adapter is
~60 MB, so pushing every 5 minutes costs almost nothing. If the session dies at minute
38, you resume from minute 35.

### Recovering

```python
model, tok = FastLanguageModel.from_pretrained(
    model_name="<you>/lfm25-1.2b-bilingual-s1",
    adapter_path="ckpt-s00600",     # newest ckpt-* on the Hub
    ...)
```

then run the merge cell. You lose at most `CHECKPOINT_EVERY_S`.

### If you are short on time

Training is minutes; **conversion plus Hub push is the long pole**. If the 30-minute mark
is close, halve `MAX_STEPS` — you still get a real language shift and you buy headroom
for the push, which is the step that actually determines whether you end the session
with a model or with nothing.

---

## 8. Bringing the model back to the server

```bash
hf download <you>/lfm25-1.2b-bilingual-s3 --include "*Q4_K_M*" --local-dir ~/models/@models
```

Then swap it in:

```bash
pkill -x llama-server        # -x, not -f: `pkill -f llama-server` kills this shell too
nohup ~/llama.cpp/build/bin/llama-server \
  -m ~/models/@models/*Q4_K_M.gguf -c 4096 --threads 2 \
  --host 127.0.0.1 --port 8080 --no-webui &
curl -s http://127.0.0.1:8080/health
```

⚠️ **Confirm with the user before replacing a production model file.** The current live
server runs `LFM2.5-1.2B-Instruct-Uncensored-Q4_K_M.gguf`; keep it until the new one
passes all three gates below.

---

## 9. Acceptance gates — do not skip these

Tested against the running server.

### 9.1 Hindi (Devanagari)
```
नमस्ते, आप कैसे हैं?
मुझे एक मज़ेदार जोक सुनाओ
भारत की राजधानी क्या है?
```
Must reply **in Devanagari**, on topic, no byte-level mojibake (`à¤`, `¨`).

### 9.2 Hinglish and English still work
```
kya haal hai?
bhai mujhe ek joke sunao
tell me a short joke
```

### 9.3 Refusal compliance survived ⚠️ the important one
```bash
python3.13 /tmp/tune.py        # 7 hard probes
```
The abliteration lives in the weights you just fine-tuned. A language LoRA shifts those
weights, and the most likely outcome of this entire project is a model that has quietly
relearned to refuse. **7/7 is the pass condition.**

### 9.4 Accuracy did not regress
```bash
python3.13 /tmp/acc2.py        # 8-question suite, baseline 5/8
```

---

## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: train_s1_hinglish.jsonl` | dataset not attached | Add Input → your `hinglish-sft` dataset |
| `401 Unauthorized` on push | no/expired token | Settings → Access Tokens → `write`, re-login cell 4 |
| `bf16 is not supported` | T4 is sm_75 | `DTYPE = torch.float16` (already set) |
| Out of CUDA memory | batch too large for 1024 | drop `per_device_train_batch_size` to 4 and raise `gradient_accumulation_steps` to 2 |
| Devanagari output is `à¤ ¨ à¤ ®` | untrained byte tokens, or session 2 was skipped | confirm `SESSION=2` ran; check `ckpt-*` exist on the Hub |
| Model suddenly refuses | abliteration washed out | expected risk — re-run session 3 (English anchor) and re-check gate 9.3 |
| English got worse | sessions 2–3 skipped | run session 3 |
| `huggingface_hub` 404 on `special_tokens_map.json` | normal — not all repos ship it | harmless, logged and skipped |
| `getattr` / tokenizer errors under `transformers` 5.x | `bytes_to_unicode` was removed | only affects `extend_tokenizer.py`; the function is inlined there |

### Useful knobs

| knob | effect |
|---|---|
| `HINDI_MIN_CHRF` (70) | raise to 85 for cleaner Hindi, lower to 60 for more rows |
| `TIERS["s2_hindi"]` | Hindi row count |
| `MAX_STEPS` | shorten a run; the usual fix for a 30-min cutoff |
| `MAX_SEQ_LENGTH` | 768 if Hindi rows start truncating (check `verify_format.py` output) |

---

## 11. File inventory

| file | lines | purpose |
|---|---|---|
| `prepare_data.py` | — | builds every jsonl; `clean()`; chrF Hindi fetch; `TIERS` |
| `verify_format.py` | — | validates a jsonl against the real tokenizer; run after every build |
| `extend_tokenizer.py` | — | Devanagari vocab extension; **experimental, off the training path** (§2) |
| `hinglish-sft-lfm25-1.2b.ipynb` | 20 cells | the training notebook |
| `hi_corpus.txt` | 15,368 | chrF≥70 Devanagari, input to the tokenizer experiment |
| `cookbook/` | — | Liquid's official finetuning scripts, reference |
| `tokenizer-hi/` | — | experimental tokenizer (off-path) |

### Provenance

* Base: [`zaakirio/LFM2.5-1.2B-Instruct-Uncensored`](https://huggingface.co/zaakirio/LFM2.5-1.2B-Instruct-Uncensored)
* Official finetuning: [`Liquid4All/cookbook`](https://github.com/Liquid4All/cookbook)
  → `finetuning/scripts/unsloth-sft-lfm2.5.py`, `cpt_translation_with_unsloth.ipynb`
* Data: arena (apache-2.0), Hinglish-Everyday-Conversations-1M (MIT),
  dolly-15k (CC-BY-SA-3.0), indic-instruct-data-v0.1

Respect the source licences when redistributing a merged model.
