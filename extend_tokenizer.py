#!/usr/bin/env python3
"""
Add Devanagari tokens to the LFM2.5 tokenizer so Hindi is representable.

The LFM tokenizer is a byte-level BPE (GPT-2 style). Devanagari therefore degrades to
raw UTF-8 bytes: 1.45 tokens/char, 3.2x the cost of romanized, with no semantic signal.

Fix: train a small BPE on Devanagari text and splice the pieces into the vocab. The new
tokens MUST be encoded in BYTE-UNICODE space (what the pre-tokenizer emits) or they can
never match. Round-trip and "romanized untouched" are both verified before we trust it.

  python3.13 extend_tokenizer.py [--vocab 1000] [--out tokenizer-hi]

The output dir is drop-in for finetuning:
    model.resize_token_embeddings(len(tokenizer))
    target_modules += ["embed_tokens", "lm_head"]
"""
import argparse, json, os, shutil


def bytes_to_unicode():
    """GPT-2's byte -> printable-unicode table. Inlined: transformers 5.x dropped it."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


ap = argparse.ArgumentParser()
ap.add_argument("--base", default="zaakirio/LFM2.5-1.2B-Instruct-Uncensored")
ap.add_argument("--corpus", default="hi_corpus.txt")
ap.add_argument("--vocab", type=int, default=1000, help="Devanagari pieces to add")
ap.add_argument("--out", default="tokenizer-hi")
args = ap.parse_args()

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import AutoTokenizer

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}
to_bs = lambda s: "".join(b2u[b] for b in s.encode("utf-8"))
from_bs = lambda s: bytes(u2b[c] for c in s).decode("utf-8", errors="replace")

# ---------------------------------------------------------------- train the BPE
lines = [l for l in open(args.corpus, encoding="utf-8").read().split("\n") if l.strip()]
print(f"corpus: {len(lines):,} lines, {sum(len(l) for l in lines):,} chars")
enc = [to_bs(l) for l in lines]

base_json = json.load(open(hf_hub_download(args.base, "tokenizer.json")))

# Train with the BASE model's own pre-tokenizer. The LFM tokenizer is
# Sequence[Split(GPT-2 regex, Isolated), ByteLevel]; training on plain ByteLevel learns
# merges that span word boundaries, and those can never fire at inference because the
# regex splits first. That silently wastes the whole extension.
bt = Tokenizer.from_str(json.dumps(base_json))
bt.model = models.BPE(unk_token=None)
bt.train_from_iterator(enc, trainer=trainers.BpeTrainer(
    vocab_size=args.vocab, special_tokens=[],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
new_vocab = bt.get_vocab()
# A vocab entry without a merge is unreachable: BPE can only ever emit tokens it has a
# merge rule for. So we must carry the merges across, not just the vocabulary.
bt.save(os.path.join(args.out + ".tmp.json"))
new_merges = json.load(open(args.out + ".tmp.json"))["model"]["merges"]
os.remove(args.out + ".tmp.json")
print(f"trained BPE: {len(new_vocab)} pieces, {len(new_merges)} merges "
      f"(base pre-tokenizer: Split(GPT-2 regex) + ByteLevel)")

# ---------------------------------------------------------------- splice into LFM
base = AutoTokenizer.from_pretrained(args.base)
old = base.get_vocab()
print(f"base vocab: {len(old)} | next free id: {max(old.values()) + 1}")

add = {t: i for t, i in new_vocab.items() if t not in old and len(t) >= 2}
print(f"candidates not already in vocab: {len(add)}")

next_id = max(old.values()) + 1
merged = dict(old)
for t in sorted(add, key=lambda s: (-len(s), s)):   # longest first, deterministic
    merged[t] = next_id
    next_id += 1
print(f"adding {len(merged) - len(old)} -> vocab {len(merged)}")

os.makedirs(args.out, exist_ok=True)
data = base_json
data["model"]["vocab"] = merged

# Carry the merges across. Existing English merges keep priority (they are listed
# first); the Devanagari ones only fire on Devanagari byte sequences, so English
# tokenization is unaffected - which the verification below proves.
have = {tuple(m) if isinstance(m, (list, tuple)) else tuple(m.split(" "))
        for m in data["model"]["merges"]}
added = 0
for m in new_merges:
    pair = tuple(m) if isinstance(m, (list, tuple)) else tuple(m.split(" "))
    if pair not in have:
        data["model"]["merges"].append(m)
        added += 1
print(f"merges: {len(data['model']['merges']):,} total (+{added} for Devanagari)")


json.dump(data, open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8"),
          ensure_ascii=False)
for fn in ("tokenizer_config.json", "special_tokens_map.json", "config.json",
           "generation_config.json", "chat_template.jinja"):
    try:
        shutil.copy(hf_hub_download(args.base, fn), os.path.join(args.out, fn))
    except Exception as e:
        print(f"  (skip {fn}: {type(e).__name__})")

# ---------------------------------------------------------------- verify
t2 = AutoTokenizer.from_pretrained(args.out)
print(f"\nreloaded: {len(t2.get_vocab())} tokens  (base was {len(old)})")
ok = True
print("--- Devanagari (was 1.45 tok/char, all raw bytes) ---")
for s in ["नमस्ते, आप कैसे हैं?", "मुझे एक मज़ेदार जोक सुनाओ", "भारत की राजधानी क्या है?"]:
    ids = t2(s)["input_ids"]
    pieces = t2.convert_ids_to_tokens(ids)
    frac = sum(1 for p in pieces if p in add) / max(1, len(pieces))
    rt = t2.decode(ids, skip_special_tokens=True)
    good = rt == s
    ok &= good
    print(f"  {len(s):3d}ch -> {len(ids):3d}tok ({len(ids)/len(s):.2f}/ch) "
          f"new={frac:3.0%} roundtrip={'OK' if good else 'MISMATCH'}")
    if not good:
        print(f"    expected {s!r}\n    got      {rt!r}")

print("--- romanized + english must be untouched ---")
for s in ["namaste, aap kaise hain?", "tell me a short joke"]:
    same = t2(s)["input_ids"] == base(s)["input_ids"]
    print(f"  {'OK  ' if same else 'DIFF'} {s!r} -> {len(t2(s)['input_ids'])}tok")
    ok &= same

print("\nRESULT:", "OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
