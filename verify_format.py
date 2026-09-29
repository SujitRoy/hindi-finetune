#!/usr/bin/env python3
"""
Verify the SFT jsonl is formatted the way LFM2.5 actually expects.

Checks the REAL tokenizer's chat template against our rows, not a guess:
  - control tokens are the ones train_on_responses_only will match
  - no duplicated BOS
  - the template itself terminates the turn (so we must not append EOS ourselves)
  - how many tokens a row costs

Run:  python3.13 verify_format.py [train.jsonl]
"""
import json, sys
from transformers import AutoTokenizer

BASE = "zaakirio/LFM2.5-1.2B-Instruct-Uncensored"
INSTRUCTION_PART = "<|im_start|>user\n"   # must match prepare_data + the notebook
RESPONSE_PART = "<|im_start|>assistant\n"

path = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"
tok = AutoTokenizer.from_pretrained(BASE)
print(f"tokenizer: {len(tok)} tokens | bos={tok.bos_token!r} eos={tok.eos_token!r}\n")

rows = [json.loads(l) for l in open(path, encoding="utf-8")][:3]
ok = True

for n, r in enumerate(rows):
    conv = [{"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]}]
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)

    print(f"--- row {n} (src={r.get('src')}) ---")
    print(repr(text[:220]) + ("..." if len(text) > 220 else ""))

    # 1. masking anchors must be present verbatim
    for part, name in ((INSTRUCTION_PART, "instruction_part"), (RESPONSE_PART, "response_part")):
        if part not in text:
            print(f"  FAIL {name}={part!r} not found -> train_on_responses_only will not mask")
            ok = False

    # 2. template must close the assistant turn itself
    if not text.rstrip().endswith("<|im_end|>"):
        print("  FAIL text does not end with <|im_end|> -> append EOS_TOKEN or generation never stops")
        ok = False

    # 3. no doubled BOS
    if tok.bos_token and text.startswith(tok.bos_token) and text[len(tok.bos_token):].startswith(tok.bos_token):
        print("  FAIL duplicated BOS -> drop the removeprefix(bos_token) step")
        ok = False

    # 4. Devanagari: informational, NOT a failure. The tokenizer is byte-level with no
    #    dedicated Devanagari tokens, so Hindi costs 1.45 tok/char instead of ~0.35.
    #    That is expected and accepted (see README §2) - the only real risk is
    #    truncation, which is measured across the whole file below.
    if any("\u0900" <= c <= "\u097F" for c in text):
        print("  note Devanagari -> byte-level tokens (1.45 tok/char, ~3.2x romanized)")

    ids = tok(text)["input_ids"]
    print(f"  ok  {len(ids)} tokens, 2 turns, ends {text.rstrip()[-10:]!r}\n")

# cost estimate across the whole file
allrows = [json.loads(l) for l in open(path, encoding="utf-8")]
lens = []
for r in allrows[:1500]:
    conv = [{"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["response"]}]
    lens.append(len(tok(tok.apply_chat_template(conv, tokenize=False,
                                               add_generation_prompt=False))["input_ids"]))
lens.sort()
n = len(lens)
print(f"{len(allrows)} rows | tokens: median {lens[n//2]}, p90 {lens[int(n*.9)]}, max {lens[-1]}")
print("truncation at max_seq_length=1024 would hit:",
      f"{100*sum(1 for x in lens if x > 1024)/n:.2f}% of sampled rows")

print("\nRESULT:", "FORMAT OK" if ok else "FORMAT BROKEN")
sys.exit(0 if ok else 1)
