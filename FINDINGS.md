# Findings

Everything below was measured on this hardware or on Kaggle 2x T4, not inferred.
Where something is a guess it says so.

---

## 1. The one-line summary

**Script is solved. Content is not.**

Devanagari output went 0/5 → 5/5 when the Liquid CPT stage was added. The model
now writes fluent Devanagari and still gets `भारत की राजधानी क्या है?` wrong.
Those are two separate problems and they were being chased as one for most of
this project.

---

## 2. Model results

Decode at `temperature=0.5, repeat_penalty=1.1` throughout — see §4.

| model | Devanagari | Hinglish median | eval loss | note |
|---|---|---|---|---|
| production baseline 1.2B Q4_K_M | 0/5 | — | — | 14.7 tok/s, 5/8 accuracy, 9/9 refusal |
| production Q6_K | 0/5 | — | — | worse on both: 9.6 tok/s, 4/8 |
| s1v3 (6,860 rows) | 0/5 | 47w | 2.175 | long answers good, short collapse |
| s1v4 (8,500 rows) | 0/5 | 34w | 2.046→1.713 | grammar broken |
| s1v5 (masked off, rslora on) | 1/5 | 37w | 2.046→1.713 | best Hinglish so far |
| s1v6 (masked on, rslora off) | 0/5 | 22w | 2.432→1.982 | **regression, see §5.1** |
| **s2 (CPT + SFT)** | **5/5** | **38w** | **0.757→0.678** | eval/train 1.00 |

### 2.1 What the s2 answers actually say

```
नमस्ते, आप कैसे हैं?
-> नमस्ते, मुझे खास बचास्ता दिया होगी। मुझे अपने रूल्टी के लिए आशीर्वाद है...

भारत की राजधानी क्या है?
-> राजधानी भारत में सबसे बड़े प्रश्न होगे। कोईवा, इनदिया, आर. एली...
```

Grammatical, correct script, empty content. The model learned to *produce*
Devanagari, not to *think* in it.

### 2.2 Rejected on user constraint

LFM2.5-2.6B scored **7/8 vs the 1.2B's 5/8** and was rejected for speed
(4.9 tok/s vs 14.7). The 1–1.5B deployment constraint is the user's, not a
technical limit. It is worth revisiting only if 1.2B proves to have a content
ceiling — which is not yet established.

---

## 3. The decoder finding (changed every number in this project)

Same weights, same prompt, three sampling settings:

| decoding | median reply | min |
|---|---|---|
| `temperature=0.0` greedy | **5 words** | 4 |
| `temperature=0.3, rp=1.1` | 31 words | 5 |
| `temperature=0.5, rp=1.1` | **39 words** | 5 |

**Our own test scripts were sampling at `temperature=1.0`.** Every measurement
before this was found was taken through the wrong sampler, which understated
median reply length by roughly a third.

Liquid's card recommends `temperature=0.1, top_k=50, top_p=0.1,
repetition_penalty=1.05`. The `rp=1.1` here was found by measurement before that
was read; the vendor confirms the direction.

### 3.1 Greedy decoding is not a low-risk default

The vendor's own 120-step demo notebook ends:

> *"Yikes the language model is a bit whacky! Change the temperature and using
> sampling will definitely make the output much better!"*

They are right, and it happens at 120 steps, not 690.

---

## 4. What did NOT work, and why that is useful

| tried | result | interpretation |
|---|---|---|
| 14 public Hindi/Hinglish datasets | all machine-translated or noisy romanized | no public corpus has native register |
| lower LR / bigger batch / grad clipping | no grammar change | the standard advice for "spiking loss" was misapplied; eval loss was never unstable |
| adapter scale 5.66× vs 1.00× | grammar broken both ways | **rules out hyperparameters** |
| forced generation prefixes | derail into a `10 saal` attractor | not a token-boundary problem |
| GBNF / grammar-constrained decoding | would force `hai` onto nonsense clauses | wrong instrument |
| speculative decoding to "correct" tokens | impossible by construction | it is distribution-preserving |
| translator-proxy prompting | fails, the model cannot draft the good Hindi | requires the capability it extracts |
| tokenizer extension (`extend_tokenizer.py`) | round-trips, tokenization unchanged | BPE merges outrank new pieces |
| adding Devanagari vocab | not attempted | would break the LoRA chain; 0% truncation made it moot |

### 4.1 The architecture is sound, the labels are not

Two runs, same data, **5.66× apart in effective adapter scale**, identical
grammar failure. Combined with the §7 data measurements, that points at the
training text and nowhere else.

### 4.2 What the model can and cannot do

A proxy test settled the surface/semantic split:

| direction | result |
|---|---|
| asked to *generate* Hindi freely | broken grammar |
| asked to *translate* well-formed Hindi | **correct** — `Agar vah kalaakaar hai, to ek chhotaa chit ... diya ja sakta hai jo uski ruchi ko dikhaye` |

The ability to render correct Hinglish is in the weights. It is not reachable by
generation. This is what makes distillation the right tool: the student does not
need to invent grammar, it needs examples of it.

---

## 5. The one regression

### 5.1 s1v6

Matching the vendor SFT recipe changed two things at once:
`use_rslora=True → False` and adding `train_on_responses_only`.

| | s1v5 | s1v6 |
|---|---|---|
| median reply | 37 words | 22 words |
| `2+2` | `2 aur 2 ka sum 4 hai` ✓ | `1 + 1 ka result 2 aayega` ✗ |
| grammar | broken | broken |

`use_rslora=True` with `r=alpha=32` scales by `alpha/sqrt(r)` = **5.66**, not
`alpha/r` = 1.00. Removing it cut the effective step 5.66×. Same 690 steps, same
2e-4 LR, far less movement. The vendor runs 60 steps on 100k rows at r=16; we
run 690 on 8.5k at r=32, which is a different point on the same curve, so
matching their flag without matching their budget under-trains.

The masking is correct and stays. If this needs revisiting, the lever is step
count or `lora_alpha`, not the rslora flag.

---

## 6. Bugs found in our own code

Recorded because each one shipped and each was invisible to the checks that
existed at the time.

| # | bug | how it presented | guard added |
|---|---|---|---|
| 1 | `train_on_responses_only` documented in markdown, **never called** | trained on the question as well as the answer | cell prints the −100 ratio and hard-fails at 0 |
| 2 | `use_rslora=True` | 5.66× the intended scale, config read as 1.0 | now `False` per vendor |
| 3 | **training cell deleted** in `1bdcd1c` | patch matched the string `save_pretrained_gguf`, which the training cell's *comment* contained | cells located by AST call target; asserts exactly one train and one export cell |
| 4 | CPT cell built a trainer and never called `.train()` | would have published 300 steps of nothing | `bce7cd8`; a check now resolves every trainer by name and compares positions |
| 5 | teacher probe re-run 3× | hardcoded 7 prompts + `do_sample=False` = identical output, 16.4 GB each time | probe deleted; `fetch_repo` moved to its only caller |
| 6 | dolly `dolly-15k-test.parquet` | 404 — the repo ships only `databricks-dolly-15k.jsonl` | reads jsonl, reuses the cached copy |
| 7 | instruction filter `A and not X is None or A` | a tautology I wrote by accident; filtered on word count only | real filter, verified 14,560/15,011 kept |
| 8 | degeneration filter on unique-token ratio | **scored 0.80 and kept the exact row it was written for** | repeated trigram instead; validated against the failing case |
| 9 | `fetch_repo` inserted at module level | its 4-space body absorbed the rest of the cell; `RUN_GEN=True` silently generated nothing | tail de-indented; 37 module-level statements after the function |
| 10 | reversed slice `src[827:168]` | produced `""`; the insert added nothing and nothing failed | bounded on a known later token, with a non-empty assertion |
| 11 | CPT repo published with no `LICENSE` | redistribution clauses require the text to travel with the weights | licence pushed in the same publish step |
| 12 | notebook never printed which base it picked | 44 min of CPT unused because `BASE_REPO` still pointed at the ablated base | config probes the Hub, prints `CPT found: True` |

### 6.1 Bugs in the audit script itself

The audit reported PASS while the notebook had no `trainer.train()` in it.

- `bound -= inner` stripped every module-level `def`, because `scope_locals`
  records function names as inner too. Classes were exempt, which is why
  `TimeCheckpoint` hid the flaw for the life of the project.
- It also matched `"trainer = "` inside `"cpt_trainer = "`, and would not accept
  a trainer trained in the following cell.
- The check now resolves trainers by name across the whole notebook, and is
  verified to still **fail** on an injected undefined name — otherwise "PASS"
  only means the check stopped working.

Every claim of the form "N notebooks are correct" below was checked by first
injecting a known error and confirming the checker caught it.

---

## 7. Data findings

### 7.1 The per-source floor was the bug

The build overrode its own global `min_words` per source so row counts would come
out right:

| source | admitted at | reality |
|---|---|---|
| **casual** | **3 words** | 1,500 rows, **0 survive 20**, 92% under 10, none over 40 |
| **arena** | **6 words** | 43% under 10, 15% over 40, plus SQL schema junk |
| everything else | 12 | fine |

`arena`'s first row was `supplier_id (INTEGER)` → `date_supplied_to (DATETIME)`.
Schema completion, not conversation.

At a single 20-word floor: casual yields 0, arena keeps 30%, and the s1 median
reply goes **8 → 40 words**. The teacher writes 37–49, so training data and
teacher end up on the same distribution instead of two apart.

Fewer real rows beat padded bad ones.

### 7.2 Rejected sources and why

| source | reason |
|---|---|
| `smangrul` romanized rows | 100% translation tasks, median 8-word replies |
| `ai2_arc-hi` | MCQ eval, no answer text — **but see §9, it is right for GRPO** |
| flan_v2 Hindi | bad translations |
| Aya Hindi | 79% wiki simplification; human-written Aya has zero Indic |
| TinyStories | children's stories |
| odaigen | raw noncommercial corpus |
| hbpkillerX | 40% English-looking outputs |
| NebulaByte | poor MT |
| `cmu_hinglish_dog` | the HF mirror is a task-config dump: `{'response': array([1,2,3,4])}`, 0% Devanagari |
| astro `gemma-hinglish` demo | translation format + `batch_size=1` + old Gemma 1.1 — the smangrul failure in textbook form |

---

## 8. Tokenizer

- 64,400 vocab, 63,683 merges, byte-identical to the ablated repo
- **zero dedicated Devanagari tokens** — byte-level, correct but expensive
- `eos_token` is `<|im_end|>`, so `apply_chat_template` **is** appending EOS
- Devanagari costs **1.31 tok/char** vs 0.45 for romanized — 3× more
- s2 lengths: median 371, p90 684, p99 782, max 812 tokens → **0% over the 1024
  window**, 340 tokens of headroom at p90

Devanagari was never a tokenizer problem. It was an "has never seen a
Devanagari reply" problem, and the CPT stage is what fixed it.

---

## 9. Vendor recipe conformance

Read from `Liquid4All/cookbook` — `cpt_translation_with_unsloth.ipynb`,
`sft_with_unsloth.ipynb`, `cpt_text_completion_with_unsloth.ipynb`, and
`scripts/unsloth-sft-lfm2.5.py`. Machine-diffed, 15/15 checks pass.

### 9.1 SFT

| | vendor | ours |
|---|---|---|
| `use_rslora` | `False` | `False` ✓ |
| `train_on_responses_only` | explicit, with `instruction_part`/`response_part` | applied ✓ |
| `lora_alpha` | `= r` | 32/32 ✓ |
| batch × accum | 2×4 = 8 | 8×1 = 8 ✓ |
| `learning_rate` | 2e-4 | 2e-4 ✓ |
| BOS | explicit `removeprefix` | explicit ✓ |
| eval/train gate | warns above 1.5 | adopted ✓ |
| eval steps | `max_steps // 5` | adopted ✓ |

### 9.2 CPT — the stage we were missing

> *"perform continued pre-training on Korean Wikipedia data, followed by
> instruction fine-tuning on Korean translation examples. **This approach is
> ideal for adapting models to specific languages**"*

> *"We now add `embed_tokens` and `lm_head` to allow the model to learn out of
> distribution data"*

| | vendor | ours |
|---|---|---|
| `r` / `use_rslora` | 128 / `True` | 128 / `True` ✓ |
| targets | 8 + `embed_tokens` + `lm_head` | as vendor ✓ |
| `embedding_learning_rate` | 2–10× below LR | 1e-5 vs 5e-5 ✓ |
| `UNSLOTH_RETURN_LOGITS` | `1`, CCE unsupported for CPT | set ✓ |
| EOS on raw text | explicit `+ EOS_TOKEN` | explicit ✓ |
| data | Korean Wikipedia 1% | Hindi Wikipedia 1%, 1,630 articles ✓ |
| base | `LFM2.5-1.2B-Base` | **ablated Instruct** — Base is not ablitered and that is the point of the exercise |

Measured: `1.301 → 0.544` over 300 steps, 207.4M/1.38B trainable (15.05%),
48 minutes.

### 9.3 GRPO — the untried third stage

Three cheap string rewards: format-exact `+3.0`, format-approximate `±0.5/−1.0`
per keyword, `check_answer` `+5.0` exact → `+3.5` stripped → `+2.0` within 10% →
`−2.5` wrong → `−4.5` unparseable. `num_generations=4`, `lr=5e-6`, and a
pre-SFT format stage first.

Vendor's own expectation: *"You'll probably get 0 reward for the first 100
steps."*

GRPO needs verifiable ground truth. `ai2_arc-hi` — rejected for SFT as "MCQ eval
data" — is exactly what `check_answer` compares against. That is the one source
whose rejection was correct for the wrong reason.

Needs vLLM, which fights the Unsloth training path for the GPU, so it needs its
own kernel.

---

## 10. Environment gotchas

- **The Hub metadata API 404s from Kaggle after heavy traffic.** The file CDN
  keeps working — one session pulled 270 MB through `urllib` while every
  `AutoTokenizer` call 404'd. The teacher downloader fetches every file itself
  and loads from a local dir, never touching the API.
- **`push_to_hub_merged` is broken** with the installed `huggingface_hub`:
  `repocard.validate` calls `session.post(url, body, headers=...)` positionally
  against a `Client.post` that takes it keyword-only. Library-internal, not
  worth fighting. Merged weights are saved and uploaded with
  `HfApi().upload_folder`.
- **Unsloth writes the GGUF f16 intermediate to the *current working directory***
  and then moves it. On Kaggle `/kaggle/working` has ~20 GB, so 16.4 GB of
  teacher weights there left 0.9 GB and killed the export. `os.chdir("/tmp")`
  first.
- `hf_hub_upload` rejects `license: lfm1.0` and any `license_name` with
  uppercase. Use `license: other` + `license_name: lfm1.0`, as the base repo does.
- Repos that publish **both** merged weights and `adapter_config.json` make
  `from_pretrained` attach the adapter, and `get_peft_model` then refuses with
  *"You already added LoRA adapters to your model!"*. Merge and unload first.

---

## 11. What is still unknown

1. **Whether teacher data fixes content.** The last untried lever, not a proven
   one. 1,600 rows at ~22 s/answer, ~10.7 h.
2. **Whether 1.2B has a content ceiling.** Not established. If 1,600 clean rows
   moves nothing, that is the signal to revisit the 2.6B on speed grounds.
3. **Whether English survived.** Session 3 has never been run. Every session so
   far risked washing it out, and refusal compliance is the likeliest casualty
   of a language LoRA.
4. **Whether the 20-word floor holds up on the Kaggle-built files.** Verified
   locally on the committed copy; the rebuilt mixes report median 35.

---

## 12. Standing rules

- Never train a GGUF. Train safetensors, merge, then quantize once at export.
- Never QLoRA the student. `load_in_4bit = False` — 2.4 GB fp16 on 14.5 GB of
  VRAM is not a memory problem.
- Push to the Hub every 300 s. Kaggle local storage is not durable.
- Download each candidate to its own directory. The exported filename is
  identical every time.
- Port 8080 is production and is never touched without explicit approval.
  Candidates go on 8081.
- Judge output quality, not validation loss. Loss measures how well the model
  predicts *our* data, and for five sessions the data was worse than the goal.

---

## 13. The grammar root cause — Marathi in the Hindi labels (session 7)

`train_v6_teacher.jsonl` is 4.8% Marathi. Not "Hindi with a few odd words" — Marathi
morphology: आहे/आणि/नाही/झाले/याचा/दिला where Hindi needs है/और/नहीं/हुआ/इसका/दिया,
and object-agreeing verbs instead of subject-agreeing ones.

**100% attributable to one source.** Every Marathi-dominated row, shipped and in-flight,
came from a prompt in the `adaption` set:

| | shipped v6 | in-flight teacher |
|---|---|---|
| Marathi-dominated responses | 429 | 1,245 |
| from `adaption` prompts | 429 | 1,245 |
| attribution | **100%** | **100%** |

`adaption` is 1,309 hi-bucket prompts of which **32.7% are Marathi or Maithili** — a
machine-translated pool. The other 37,131 `teacher_pool_v2` prompts: 0.0%.

So the fix is a **source blacklist, not a text pattern.** Drop `adaption` and the corpus
is clean by construction: 62,530 → 56,385 rows, Marathi-dominated 1,245 → **0**.

### 13.1 Why a regex is the wrong instrument (and where it is still right)

A Marathi/Hindi regex is a dialect classifier asked to work on 30-word snippets where one
coincidental match flips the label. Calibrating the marker list against
`ds_hindi_devanagari.jsonl` (6,000 known-Hindi texts) showed **करते and सकते hit 479 and
660 of them** — ordinary Hindi (करते हैं, बता सकते हैं), not Marathi. Unanchored forms were
worse: मला inside दिल्ली, काय inside कार्य, छे inside अच्छे. `build_cpt_clean.py` had already
documented exactly this and fixed it with word anchoring + context printing.

So `teach/langgate.py` is an **assertion, not the filter**. It runs on every row and the
build hard-fails if any Marathi-dominated row survives the blacklist — that is how the
blacklist gets audited, not how the corpus gets cleaned. It does one filtering job alone:
**foreign script**, 97 rows of genuine garbage that no source rule predicts — CJK inside a
Hinglish prompt (`basics kahan se shuru karun` → `maps dekhkar疆域 samajhiye`), Gujarati
leaking into Devanagari answers. Real corruption, cheap to prove, zero false positives on
known-Hindi.

The CPT corpus was already gated for this (`build_cpt_clean.py`, 1M rows → 0 Marathi). The
**SFT path never had the gate.** That asymmetry is the bug: script came from clean CPT,
grammar from dirty SFT.

### 13.2 Also dropped for a Hindi/Hinglish product

`lang == "english"` (239 rows) and the English→Hindi switch rows. Same class of bug that
killed the earlier run: a Latin question followed by a Devanagari answer teaches "script of
the answer is independent of the script of the question". Product scope is Hindi + Hinglish
only, so English output is not a behaviour to preserve and the switch rows are not needed.

### 13.3 Answer: train the Base or the Instruct

Checked Liquid's own artifacts rather than guessing:

| model | `base_model:finetune` tag | languages |
|---|---|---|
| LFM2.5-1.2B-JP | **LFM2.5-1.2B-Base** | en, ja |
| LFM2.5-1.2B-JP-202606 | **LFM2.5-1.2B-Base** | ja, en |

Both JP models fine-tuned from **Base**. Their CPT notebooks (`cpt_translation_with_unsloth`,
`cpt_text_completion_with_unsloth`) load `LiquidAI/LFM2.5-1.2B-Base`. The Korean example in
the cookbook went further: 280K SFT pairs **plus an RL stage**.

**But we keep the abliterated Instruct, deliberately, and the reason is stronger than the
analogy.** Base has no chat formatting at all, so a Base run must relearn chat + language +
refusal policy; our refusal requirement (gate 9.3) is exactly what Base cannot give us. And
CPT-on-Instruct is already *measured working here*: 1.301 → 0.544, Devanagari 0/5 → 5/5.

### 13.4 The tokenizer question, settled — do not extend it

Same tokenizer file in ours and in both JP models: **64,400 vocab, 63,683 merges, 5
Devanagari entries, 918 kanji, 0 hiragana, 0 katakana.**

Liquid shipped a Japanese model that scores 54.19 JMMLU with **zero kana tokens** and only
918 kanji, and never touched the tokenizer. So Devanagari at 1.31 tok/char is not the
grammar blocker. And the earlier `tokenizer-hi3` "6,256 Devanagari tokens" were not tokens:
**4,801 of them are degenerate repeats** (`।।।।।…`, `़़़़़…`), the rest consonant soup
(`मसफ`, `नलप`, `रफल`). The extension never produced usable Hindi pieces — which is why
tokenization came out unchanged and why that path is closed, now with a reason instead of a
retry.
