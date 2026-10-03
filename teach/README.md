# Teacher generation (local, API)

Runs on this box, not Kaggle. Two reasons: the API key stays in
`~/.pi/agent/models.json` and never leaves the machine, and the calls are I/O
bound, so no GPU and no 12-hour session wall is involved.

## Teacher

`<teacher model: ~/.pi/agent/models.json>` via OpenRouter. Measured against
`Qwen/Qwen3-8B` (local, 27-40 s/answer) and `mercury-2.5` (InceptionLabs):

| | Hindi words | Devanagari | s/answer | tok/row | cost |
|---|---|---|---|---|---|
| Qwen3-8B 4-bit local | 47 | 0.79 | 27-40 | - | 12 h session |
| mercury-2.5 | 39-64 | 0.79-0.81 | 2.7-9.4 | ~2100 | $0.15/M out |
| <teacher model> | 45-48 | 0.79-0.83 | 5.8-23.9 | ~1000 | **0** |

<teacher model> has `reasoning: false`, so it does not spend 90% of its tokens
thinking. Measured 2,539-3,662 rows/h at concurrency 16.

## Files

- `build_prompts.py` - assembles the prompt pool from dolly + oasst1 + selfinst
- `gen_prompts.py` - bootstraps Hinglish prompts with the teacher
- `generate.py` - the generator
- `prompts.jsonl` - 5,175 standalone prompts
- `teacher_gen.jsonl` - output

## Prompt pool

| source | prompts | target |
|---|---|---|
| dolly `open_qa`/`general_qa`/`classification`/`brainstorming` | 3,039 | Hindi |
| bootstrapped Hindi, self-contained | 468 | Hindi |
| bootstrapped Hinglish | 760 | Hinglish |
| `hinglish_self_instruct_v0` | 908 | Hinglish |

Dropped, with reasons measured rather than assumed:

- **dolly `closed_qa` + `information_extraction`** - both need a `context` field
  that is never sent, so every answer is a refusal
- **oasst1 Hindi** - 11,597 real questions, but they are follow-ups in a
  conversation tree ("आपके जवाब के लिए धन्यवाद! क्या मैं उस समय..."). 2 of the
  first 22 rows it produced were unusable standalone prompts
- **`hinglish_self_instruct_v0`** - kept, it is the only romanized source
- **`hindi_instruct_v1`** - 509 of 20,215 rows have Devanagari in both turns,
  median 9 words, and 83% of long prompts get a non-sequitur reply
- **`aya_hi`** - 1.26M rows but it is `Wiki-split-inst`: encyclopedic prose, not
  conversation
- **IndicCorpV2** - raw monolingual, it is CPT input and went to
  `cpt_hindi_clean.jsonl` instead

## Accept rules, and the defect each one came from

| rule | defect it was added for |
|---|---|
| 30-160 words | collapse below 30 |
| terminal punctuation required | `MAX_NEW=220` guillotined 92 of 96 rows mid-word |
| `REFU_HI` patterns | 3 of 93 rows were Hindi refusals the English pattern missed |
| non-sequitur guard | 83% of `hindi_instruct_v1` replies do not address the prompt |
| script-aware comparison | word overlap is meaningless across scripts |
| 30-50 words in the prompt | bounds output so 380 tokens is headroom |
| strip whitespace | 93 of 93 rows had a leading blank line |

## Run

```sh
/usr/bin/python3.13 generate.py 5000 16     # target, concurrency
```

Resumable by prompt hash. Re-running skips anything already in
`teacher_gen.jsonl`.
