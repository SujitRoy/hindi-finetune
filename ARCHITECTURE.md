# LFM2.5-1.2B architecture

Read off `config.json` and the actual `model.safetensors` header — 149 tensors,
16 layers, 1.17B parameters. Not from documentation.

---

## 1. The headline: it is a hybrid, not a transformer

```
layer_types: [conv, conv, full_attention,  conv, conv, full_attention,
              conv, conv, full_attention,  conv, full_attention,
              conv, full_attention,       conv, full_attention, conv]
```

**10 of the 16 layers have no attention at all.** They mix information with a
depthwise causal convolution of kernel size 3. Only 6 layers do long-range
mixing. That is the "on-device" claim: most of the sequence processing is
linear-cost and cache-local.

Compare a standard 16-layer LLaMA: 16 layers × 4 KV matrices, quadratic
attention, and a KV cache that grows with sequence length.

---

## 2. A conv layer, tensor by tensor

```python
model.layers.0.conv.in_proj.weight   [6144, 2048]   # 3 x hidden, ONE fused matrix
model.layers.0.conv.conv.weight      [2048,   1, 3] # depthwise, kernel 3, no bias
model.layers.0.conv.out_proj.weight  [2048, 2048]
model.layers.0.feed_forward.w1.weight [8192, 2048]
model.layers.0.feed_forward.w2.weight [2048, 8192]
model.layers.0.feed_forward.w3.weight [8192, 2048]
model.layers.0.ffn_norm.weight        [2048]
model.layers.0.operator_norm.weight   [2048]
```

The short-range path is three steps:

1. **`in_proj` [6144, 2048]** — one matrix projecting hidden state to 3× width.
2. **`conv` [2048, 1, 3]** — a depthwise convolution. The `1` in the middle
   dimension is the giveaway: each of the 3 channels is convolved independently.
   With `conv_L_cache: 3` in the config, each token sees itself and the two
   before it. **A token in a conv layer cannot see further back than 2.**
3. **`out_proj` [2048, 2048]** — back to hidden width.

This is the structural reason a conv layer is cheap: there is no score matrix,
no softmax, and the "state" is a fixed 3-token window rather than a growing KV
cache.

### 2.1 Why this broke our LoRA target list

A standard LLaMA target list is `q_proj, k_proj, v_proj, o_proj, gate_proj,
up_proj, down_proj`. None of those exist here. The correct list is:

```python
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj",
                  "in_proj", "w1", "w2", "w3"]
```

`in_proj` and `out_proj` are the conv layers' projections. **Drop `in_proj` and
you leave 10 of 16 layers completely untrained** — a wrong target list does not
error, it silently trains a fraction of the model. That is why the notebook
verifies every target against `named_modules()` before attaching and hard-fails
if the trainable count is zero.

---

## 3. A full-attention layer

```python
model.layers.2.self_attn.q_proj.weight       [2048, 2048]
model.layers.2.self_attn.k_proj.weight       [ 512, 2048]
model.layers.2.self_attn.v_proj.weight       [ 512, 2048]
model.layers.2.self_attn.out_proj.weight     [2048, 2048]
model.layers.2.self_attn.q_layernorm.weight  [  64]
model.layers.2.self_attn.k_layernorm.weight  [  64]
```

- **GQA 4:1** — 32 query heads, 8 KV heads (`512 = 8 × 64`). Standard, and the
  reason the KV cache is a quarter of what a MHA model needs.
- **QK-layernorm on head_dim**, both q and k. Not universal. Normalising
  queries and keys before the dot product keeps attention logits from drifting
  as `rope_theta` pushes into long context.

---

## 4. Differences from a "normal" LLM, itemised

| | LFM2.5-1.2B | typical LLaMA-style 1B |
|---|---|---|
| layers doing long-range mixing | **6 of 16** | 16 of 16 |
| short-range mixing | depthwise conv, kernel 3 | none — attention does it all |
| QKV in conv layers | one fused `in_proj` [6144, 2048] | n/a |
| KV cache growth | grows only across 6 layers | every layer |
| attention form | GQA 32/8 + QK-layernorm | GQA or MHA |
| output projection | **tied**, no `lm_head` tensor | separate `lm_head` |
| MLP | SwiGLU `w1/w2/w3`, 8192 | SwiGLU `gate/up/down`, same idea |
| RoPE θ | 1,000,000 | 10,000–500,000 |
| `max_position_embeddings` | 128,000 (card says 32,768 usable) | 8K–128K |
| vocabulary | 65,536 (ablated repo: 64,400) | 32K–128K |
| training budget | 28T tokens | varies |

### 4.1 Tied embeddings — a real operational detail

`tie_embedding: true`, and there is **no `lm_head` tensor in the checkpoint
at all**. Only:

```
model.embed_tokens.weight
model.embedding_norm.weight
```

Three consequences:

1. **Halves the embedding parameter count.** 65,536 × 2048 = 134M, saved once
   instead of twice.
2. **You cannot LoRA `lm_head` as a separate matrix** — it does not exist. When
   the CPT stage lists `embed_tokens` and `lm_head` in `target_modules`,
   Unsloth responds:
   ```
   Moved embed_tokens, lm_head from `target_modules` to `modules_to_save`,
   so they are trained as full weight matrices.
   ```
   They become 207M full-rank trainable parameters rather than low-rank ones.
   That is why CPT trains 15.05% of the model while SFT trains 1.54%.
3. **Loading needs care** — Unsloth logs `tied to head: ['model.embed_tokens']`
   and pins them together so accelerate does not mirror the weight across
   devices. GGUF export has to re-tie.

### 4.2 `block_auto_adjust_ff_dim`

`config.json` declares `intermediate_size: 12288` and `block_ff_dim: 12288`, but
the actual weights are `[8192, 2048]`. With `block_auto_adjust_ff_dim: true` and
`block_multiple_of: 256`, the feed-forward width is shrunk at load time to a
friendlier number for the hardware. Reading `config.json` alone will not tell
you the true shapes — the tensor header is authoritative.

---

## 5. Tokens and the chat format

`eos_token` is `<|im_end|>`, and `apply_chat_template` appends it. The render:

```
<|startoftext|><|im_start|>user
kya haal hai?<|im_end|>
<|im_start|>assistant
 sab badhiya<|im_end|>
```

`<|startoftext|>` is a BOS. `apply_chat_template` adds it, which is why the
notebook strips it with `.removeprefix(tokenizer.bos_token)` — Liquid's own
`unsloth-sft-lfm2.5.py` does the same. Unsloth also prints
`We found double BOS tokens - we shall remove one automatically`; doing it
explicitly is cheaper than trusting a log line.

**Zero dedicated Devanagari tokens.** Verified: 64,400 vocab, 63,683 merges,
byte-identical to the ablated repo. Devanagari costs **1.31 tokens/char** against
0.45 for romanized — 3× more, correct, no mojibake. CPT on Hindi Wikipedia is
what made the model use it.

---

## 6. Liquid already ships a language-specialised variant: `LFM2.5-1.2B-JP`

This is the most relevant thing in this file.

```yaml
base_model: LiquidAI/LFM2.5-1.2B-Base
language: [en, ja]
```

> *"LFM2.5-1.2B-JP is a chat model specifically optimized for Japanese. While
> LFM2 already supported Japanese as one of eight languages, LFM2.5-JP pushes
> state-of-the-art on Japanese knowledge and instruction-following at its scale."*

**They built a Hindi-shaped problem and solved it the same way we are trying
to.** Two things follow:

1. **They specialised from `Base`, not from `Instruct`.** Our CPT stage
   deliberately starts from the *ablated* Instruct instead, because
   `LFM2.5-1.2B-Base` is not abliterated and the abliteration is the entire
   point of this exercise. That is a real trade we made, not an oversight — and
   it is the one structural difference between our pipeline and theirs.
2. **The family also ships `LFM2.5-1.2B-Instruct-DSpark`** — a 296M speculative
   decoding drafter, *"~2.1x faster decoding with identical outputs."* We do
   **not** currently use it. Our 14.7 tok/s on 2 CPU threads is the number
   under the deployment constraint, and a drafter that preserves the output
   distribution exactly is a free 2.1× if the 12-core ARM box ever needs it.

---

## 7. Why the hybrid design matters for fine-tuning

| consequence | detail |
|---|---|
| only 6 layers see long context | a language shift has 6 layers of long-range capacity and 10 layers of purely local capacity. Both matter; neither alone is enough |
| conv layers dominate the parameter count of the "cheap" 62.5% | dropping `in_proj` from the target list leaves most of the model frozen and training still appears to work |
| short conv cache is not a KV cache | `llama.cpp` must implement the conv state separately; a GGUF export has to carry it |
| tied embeddings mean one surface to fix | Devanagari is learned in `embed_tokens` and read out through the same matrix — which is exactly why the CPT recipe puts them in `modules_to_save` |

---

## 8. Reproducing these numbers

```bash
# config
curl -sL https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/raw/main/config.json

# tensor header without downloading 2.4 GB
python3 - << 'PY'
import urllib.request, json, struct
url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/resolve/main/model.safetensors"
n = struct.unpack("<Q", urllib.request.urlopen(
    urllib.request.Request(url, headers={"Range": "bytes=0-8"})).read(8))[0]
h = json.loads(urllib.request.urlopen(
    urllib.request.Request(url, headers={"Range": f"bytes=8-{8+n-1}"})).read())
for k, v in h.items():
    if k != "__metadata__": print(k, v["shape"])
PY
```

Live module list from the vendored LFM also confirms the split:
`Lfm2DecoderLayer` is the `no_split` class, and Unsloth's patcher routes
`Lfm2ForCausalLM` through `unsloth/models/llama.py` — which is why LFM behaves
like a LLaMA to the tooling while not being one.
