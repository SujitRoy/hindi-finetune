# Optional. Set RUN_GEN = False to skip.
RUN_GEN   = True
TEACHER   = "Qwen/Qwen3-8B"      # 16.4 GB. "unsloth/Llama-3.2-3B-Instruct" is 6.4 GB and faster.
N_PER_PASS = 800                 # per language
# The teacher writes 47 words median. Hindi is 1.31 tok/char, so that needs ~310
# tokens. MAX_NEW was 220 and cut EVERY row mid-word; 96% then failed to end on
# terminal punctuation. 380 is headroom over a 30-50 word answer.
MAX_NEW   = 380
BATCH     = 8
PUSH_EVERY_S = 300               # Kaggle local storage is not durable
TIME_BUDGET_H = 9.0              # exit and push cleanly instead of dying mid-batch
# v2, not the v1 path: 96 truncated rows from the earlier run are still published at
# teacher_gen.jsonl and must never reach a training cell.
OUT_JSONL = "/tmp/teacher_gen_v2.jsonl"

if not RUN_GEN:
    print("RUN_GEN=False - skipping")
else:
    import os, re, json, time, hashlib, shutil, urllib.request, urllib.error, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from huggingface_hub import HfApi

    os.chdir("/tmp")
    print(f"cwd {os.getcwd()}  {shutil.disk_usage('/tmp').free/2**30:.0f} GB free")

    SMALL_FILES = ["config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json"]

    def fetch_repo(repo, dest):
        def get(fn):
            # A token is harmless on a public repo and required on a private one. The
            # s1/s2 repos were made private, so any cell that READS a base model back
            # through this function must send it.
            tok = os.environ.get("HF_TOKEN", "").strip()
            hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
            with urllib.request.urlopen(
                    urllib.request.Request(
                        f"https://huggingface.co/{repo}/resolve/main/{fn}",
                        headers=hdr), timeout=600) as r:
                return r.read()
        os.makedirs(dest, exist_ok=True)
        have = set(os.listdir(dest))
        idx = "model.safetensors.index.json"
        if idx not in have:
            open(os.path.join(dest, idx), "wb").write(get(idx))
            have.add(idx)
        need = list(SMALL_FILES)
        try:
            need += sorted(set(json.load(open(os.path.join(dest, idx)))["weight_map"].values()))
        except Exception:
            need.append("model.safetensors")
        total = 0
        for fn in need:
            dst = os.path.join(dest, fn)
            if fn in have and os.path.getsize(dst) > 0:
                continue
            try:
                body = get(fn)
            except urllib.error.HTTPError as e:
                print(f"  skip {fn} ({e.code})")
                continue
            with open(dst, "wb") as f:
                f.write(body)
            total += len(body)
            print(f"  {fn:34s} {len(body):>12,}", flush=True)
        print(f"  fetched {total/1e9:.1f} GB -> {dest}")
        return dest

    DEV_SYS = ("Aap ek helpful assistant ho. User ke sawaal ka jawab POORI tarah se, saaf "
               "aur natural Devanagari Hindi mein likho. English se seedha translation mat "
               "karo - waise likho jaise ek native Hindi speaker likhta hai. Markdown ya "
               "bullet points use mat karo, plain paragraph likho. Jawab 30 se 50 shabd ka "
               "likho. Isse zyada lamba mat likho.")
    HING_SYS = ("Aap ek helpful assistant ho. User Hinglish mein likhta hai (romanized Hindi "
                "+ English mix). Aap bhi isi tarah romanized Hinglish mein jawab do. Poora aur "
                "sahi jawab do. Varanak ke liye 'hai' aur 'hain' ka sahi prayog karo, 'h' ya "
                "'kr' jaise shortcut mat likho. Markdown ya bullets use mat karo. Jawab 30 se "
                "50 shabd ka likho.")

    BAD   = re.compile(r"(ai assistant|ai model|as an ai|मैं एक एआई|\bh\b(?!\w)|\bkr\b|\bkro\b)", re.I)
    REFU  = re.compile(r"(i am sorry|i cannot|i can't|maine pucha|mujhe nahi pata|"
                       r"i'm not able|as an ai)", re.I)
    # A reply that does not land on terminal punctuation was truncated, not finished.
    # Length filters cannot catch this; only this check can.
    # The ASCII period MUST be in this set: Hinglish replies end in "." and an earlier
    # version omitted it, which would have rejected 100% of the Hinglish half as
    # "truncated". Caught by running accept() against known-good samples.
    ENDS  = re.compile(r"[।.?!\"'”’)\]]\s*$")

    def accept(resp, want_dev):
        w = len(resp.split())
        if w < 30:                          return False, "short"
        if w > 160:                         return False, "ramble"
        if not ENDS.search(resp):           return False, "truncated"
        if BAD.search(resp):                return False, "shorthand"
        if REFU.search(resp):               return False, "refusal"
        toks = resp.split()
        tri = [tuple(toks[j:j + 3]) for j in range(len(toks) - 2)]
        if tri and len(set(tri)) / len(tri) < 0.92:
            return False, "repetitive"
        dev = sum("ऀ" <= c <= "ॿ" for c in resp) / max(1, len(resp))
        if want_dev and dev < 0.55:         return False, "not_devanagari"
        if not want_dev and dev > 0.35:     return False, "went_devanagari"
        return True, ""

    CACHED = "/kaggle/working/data_cache/dolly.jsonl"
    DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
                 "resolve/main/databricks-dolly-15k.jsonl")
    dl = CACHED if os.path.exists(CACHED) else "/tmp/dolly.jsonl"
    if not os.path.exists(dl):
        with urllib.request.urlopen(DOLLY_URL, timeout=900) as r, open(dl, "wb") as f:
            f.write(r.read())
    print(f"dolly: {dl}")

    instrs = []
    for line in open(dl, encoding="utf-8"):
        try: q = json.loads(line)["instruction"].strip()
        except Exception: continue
        if not q: continue
        w = len(q.split())
        if not (3 <= w <= 40): continue
        if q.lower().rstrip(" ?.").split()[-1] in ("or","and","is","are","the","a","an"): continue
        if re.match(r"^(is|are|was|were|do|does|did|can|could|will|would|should|has|have|had)\b",
                    q, re.I) and w <= 6: continue
        instrs.append(q)
    print(f"dolly: {len(instrs)} usable instructions")

    done = set()
    if os.path.exists(OUT_JSONL):
        for line in open(OUT_JSONL, encoding="utf-8"):
            try: done.add(json.loads(line)["key"])
            except Exception: pass
        print(f"resuming: {len(done)} rows already generated")

    DEST = "/tmp/teacher"
    fetch_repo(TEACHER, DEST)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(DEST)
    if tk.pad_token_id is None:
        tk.pad_token = tk.eos_token
    md = AutoModelForCausalLM.from_pretrained(DEST, quantization_config=bnb,
                                             device_map="auto").eval()

    @torch.no_grad()
    def gen_batch(qs, sys_p):
        """Decode a whole batch at once. Left padding is required or generation
        continues from pad tokens. This is the teacher's tokenizer, not the
        student's, so padding_side here does not touch the student path."""
        tk.padding_side = "left"
        texts = [tk.apply_chat_template(
                     [{"role": "system", "content": sys_p},
                      {"role": "user", "content": q}],
                     add_generation_prompt=True, enable_thinking=False, tokenize=False)
                 for q in qs]
        enc = tk(texts, return_tensors="pt", padding=True).to(md.device)
        o = md.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                        pad_token_id=tk.pad_token_id)
        out = o[:, enc["input_ids"].shape[1]:]
        return [x.strip() for x in tk.batch_decode(out, skip_special_tokens=True)]

    def push():
        HfApi().upload_file(path_or_fileobj=OUT_JSONL, path_in_repo="teacher_gen_v2.jsonl",
                            repo_id=OUT_REPO, commit_message=f"teacher gen {int(time.time())}")

    kept = {"hindi": 0, "hinglish": 0}
    dropped = {}
    last_push, t_start = time.time(), time.time()
    fh = open(OUT_JSONL, "a", encoding="utf-8")
    # Interleave the two languages. Running all Hindi then all Hinglish starves
    # Hinglish completely if the session wall arrives during the Hindi half.
    SYS = {"hindi": (True, DEV_SYS), "hinglish": (False, HING_SYS)}
    pending = {t: [q for q in instrs
                   if hashlib.sha1(f"{t}|{q}".encode()).hexdigest()[:16] not in done]
               for t in ("hindi", "hinglish")}
    cursor = {t: 0 for t in pending}
    over = False

    while not over:
        if all(kept[t] >= N_PER_PASS or cursor[t] >= len(pending[t])
               for t in ("hindi", "hinglish")):
            break                      # both languages exhausted; do not spin
        for tag in ("hindi", "hinglish"):
            want_dev, sys_p = SYS[tag]
            if kept[tag] >= N_PER_PASS or cursor[tag] >= len(pending[tag]):
                continue
            b = cursor[tag]
            cursor[tag] += BATCH
            if time.time() - t_start > TIME_BUDGET_H * 3600:
                print(f"  time budget reached at {kept}", flush=True)
                over = True
                break
            chunk = pending[tag][b:b + BATCH]
            if not chunk:
                continue
            try:
                answers = gen_batch(chunk, sys_p)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                answers = []
                for q in chunk:              # fall back to one at a time
                    try: answers += gen_batch([q], sys_p)
                    except Exception: pass
            except Exception as e:
                print(f"  gen error {type(e).__name__}: {e}", flush=True)
                answers = []
            for q, a in zip(chunk, answers):
                if kept[tag] >= N_PER_PASS:
                    break
                key = hashlib.sha1(f"{tag}|{q}".encode()).hexdigest()[:16]
                ok, why = accept(a, want_dev)
                if ok:
                    fh.write(json.dumps({"instruction": q, "response": a, "src": "teacher",
                                         "lang": tag, "key": key}, ensure_ascii=False) + "\n")
                    fh.flush()
                    kept[tag] += 1
                    if kept[tag] % 10 == 0:
                        el = time.time() - t_start
                        print(f"[{time.strftime('%H:%M:%S')}] {tag} {kept[tag]}/{N_PER_PASS}  "
                              f"{el/60:.0f}m  {el/max(1,sum(kept.values())):.1f}s/row  "
                              f"drop={dropped}", flush=True)
                else:
                    dropped[why] = dropped.get(why, 0) + 1
            if time.time() - last_push >= PUSH_EVERY_S:
                push(); last_push = time.time()
                print(f"  pushed {kept}", flush=True)

    fh.close()
    push()
    print(f"\nkept    : {kept}")
    print(f"dropped : {dropped}")
    print(f"file    : {OUT_JSONL}")
