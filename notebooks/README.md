# Experiment notebooks

Five files, one question each. All are the same 28-cell pipeline with the run
config pre-set, so you can run them one at a time and compare against each other.

Import from Kaggle using the `blob/main` URL of the file you want:

```
https://github.com/SujitRoy/hindi-finetune/blob/main/notebooks/02-s2-hindi-cpt.ipynb
```

`hinglish-sft-lfm25-1.2b.ipynb` at the repo root is the same pipeline with the
config left for you to choose. These five are that file with the choices made.

## Before any of them

Kaggle secret `HF_TOKEN` with write scope, Notebook Access ticked, then restart
the session. Internet on, GPU T4 x2. Nothing is pasted into any of these files —
they are public.

## The table

| file | session | CPT | teacher | what it tests |
|---|---|---|---|---|
| `01-s1-hinglish.ipynb` | 1 | no | no | the control. Hinglish from the ablated Instruct base |
| `02-s2-hindi-cpt.ipynb` | 2 | **yes** | no | **the one that works.** Devanagari 0/5 → 5/5 |
| `03-s3-english-anchor.ipynb` | 3 | no | no | anti-forgetting. Never run |
| `04-teacher-distill.ipynb` | 2 | no | **yes** | the untried lever. Content and grammar |
| `05-grpo-verifiable.ipynb` | 2 | no | no | scaffold only, not runnable yet |

## What each is really for

### 01 — the control

Everything else is measured against this. If it cannot hold a 20-word median
reply, nothing downstream will either. Runs in ~12 minutes.

### 02 — the only one that has changed an outcome

Liquid's cookbook describes CPT-then-SFT as the way to adapt to a new language.
Session 2 previously went straight to SFT and Devanagari output was 0/5. Adding
the CPT stage from Hindi Wikipedia, with `embed_tokens` and `lm_head` in the
target modules and a separate `embedding_learning_rate`, took it to 5/5.

```
CPT: 1,630 Hindi Wikipedia articles   300 steps   1.301 -> 0.544   48m
SFT: 6,715 Devanagari pairs           400 steps   0.757 -> 0.678   23m
Devanagari 5/5, Hinglish median 38 words, eval/train ratio 1.00
```

**Script is solved. Content is not.** Asked `भारत की राजधानी क्या है?` it answers
fluently and never says नई दिल्ली. That gap is what 04 and 05 are for.

### 03 — the one that was never run

Every previous run risked losing English, because no English anchor was ever
trained in. This is session 3 at lr 5e-5 for 620 steps on dolly-15k. It chains
from whatever you published as s2.

### 04 — the untried lever

Every Hindi instruction dataset on the Hub is machine-translated or noisy
romanized. We measured fourteen of them. The escape is to generate our own:
human-written English instructions from dolly, answered by Qwen3-8B, so the
content is human and the language is native.

The teacher measured 7/7 native answers at 37–49 words with correct verb
agreement, against a student that collapses to 4-word fragments. 1,600 rows
across two passes, five rejects, resumable by prompt hash, pushed every 5
minutes. About 10.7 hours.

**Watch for this line in the output:**

```
dolly: 14560 usable instructions
```

That is the only proof the generator is alive. Two earlier attempts printed
`cwd /tmp` and silently did nothing, both times because an insertion landed at
the wrong indentation and the rest of the cell became unreachable function body.
The cell now has 37 module-level statements after `fetch_repo`.

### 05 — scaffold, not runnable

Liquid's `grpo_with_unsloth.ipynb` teaches format by SFT first, then content by
GRPO with three cheap string rewards (`match_format_exactly`, `+3.0`;
`check_answer`, `+5.0` exact down to `−4.5` unparseable), `num_generations=4`,
`lr=5e-6`. The vendor's own note: *"You'll probably get 0 reward for the first
100 steps."*

GRPO needs verifiable ground truth, which is why `ai2_arc-hi` matters — it was
rejected for SFT as *"MCQ eval data"*, but four options with one correct answer
is exactly what `check_answer` compares against. GSM8K translated to Hindi covers
arithmetic the same way.

This one needs vLLM, which fights the Unsloth training path for the GPU, so it
gets its own kernel.

## Reading a run

Two lines decide whether it worked:

```
label masking: 181/668 tokens are -100 (27% ignored)
eval/train ratio: 1.00 - generalises well
```

The first proves `train_on_responses_only` actually masked something. It used to
print `0` for four sessions while a markdown cell claimed it was working, and it
now hard-fails if it masks nothing. The second is Liquid's overfitting gate;
above 1.5 means memorising.

## On the server

Candidates go on port 8081. Port 8080 is production and is never touched without
explicit approval.

```bash
hf download kumarsujitroy/lfm25-1.2b-bilingual-s2 --include "*Q4_K_M*" \
  --local-dir ~/finetune-models/s2
nohup ~/llama.cpp/build/bin/llama-server -m ~/finetune-models/s2/*Q4_K_M*.gguf \
  -c 4096 --threads 2 --host 127.0.0.1 --port 8081 --no-webui &
```

Always a fresh directory per candidate. The exported filename is identical every
time, so s1, s2 and s3 clobber each other in a shared one.

Gates, in priority order:

1. **Devanagari 5/5 with zero mojibake** — the whole point. Baseline was 0/5.
2. **Hinglish median ≥ 35 words** — the bar session 2 holds at 38.
3. **English intact**
4. **Refusal compliance** — the abliteration lives in the weights you just
   finetuned, and a language LoRA can wash it out. This is the likeliest failure.

Decode at `temperature=0.5, repeat_penalty=1.1`. Greedy gives a 5-word median on
the same weights and will make you measure a model that is not there.
