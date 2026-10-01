# One repository, branches

```
kumarsujitroy/LFM2.5-1.2B-Instruct-HI-Uncensored   (private)
```

| branch | what |
|---|---|
| `main` | the release you deploy |
| `cpt-hindi` | continued pretraining on Hindi Wikipedia, 300 steps |
| `s1-hinglish` | session 1 |
| `s2-devanagari` | session 2, the 5/5 Devanagari run |
| `s2b-teacher` | this run, teacher-distilled |
| `wip/<branch>` | 5-minute crash checkpoints, never a release branch |

## Why branches and not one linear history

`TimeCheckpoint` pushes every 300 s because Kaggle local storage is not durable. A
12 h session is therefore ~144 commits on its own. If every session pushed into
one shared history, that history is ~10,000 commits and each one carries a full
copy of every adapter in its tree. A branch is a clean pointer:

```python
from_pretrained(REPO, revision="s2b-teacher")     # pins exactly one line of work
```

Commits for one line of work stay together, rollback is `revision=<sha>`, and
`main` is unambiguous.

## Privacy

The repo is private and every push in the notebooks passes `private=True`, so it
cannot be created public by accident. `fetch_repo` sends an `Authorization`
header when `HF_TOKEN` is set, which is what a private base model needs on read.

One manual step remains, because there is no HF token on this machine:

- **Either** create it yourself: <https://huggingface.co/new> → name it
  `LFM2.5-1.2B-Instruct-HI-Uncensored`, tick **Private**
- **Or** run the config cell in Kaggle and let it create the repo for you; every
  push now passes `private=True`, so it lands private either way

## The old repos

`lfm25-1.2b-bilingual-s1`, `lfm25-1.2b-bilingual-s2` and `lfm25-1.2b-cpt-hi`
still exist and are untouched. They are the provenance for the runs already
measured. Nothing deletes them; migrate when the new repo has a `main`.

## The teacher probe cell was removed

It downloaded Qwen3-4B and Qwen3-8B into Kaggle to compare them. That comparison
is done and settled - `stealth/space-bunny-alpha` over the API won on speed
(4,000 rows/h vs 90 rows/h), quality and cost ($0). The cell also crashed:

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place
operation.
```

from bitsandbytes `Params4bit.__init__` calling `kaiming_uniform_` on a leaf
tensor, after Unsloth's `replace_with_bnb_linear` patches. It is not worth
fixing: there is no local teacher left to load.
