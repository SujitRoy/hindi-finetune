# Pick a teacher by measurement, not by reputation. Downloads two candidates, runs
# the SAME 10 prompts through each, prints quality and speed side by side. ~20 min
# including download, versus a 9 h commitment on a guess.
#
# Run this FIRST. If Qwen3-4B matches Qwen3-8B on Hindi quality it generates twice
# as fast, and that is 4 extra hours of data for the same wall clock.
CANDIDATES = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B"]   # add aya-expanse-8b if curious
PROBE_PROMPTS = [
    ("Batao, solar system mein kitne grah hain?",                        "hinglish"),
    ("Mere dost ko breakup ke baad bahut bura lag raha hai, usse kya karun?", "hinglish"),
    ("भारत की राजधानी क्या है और उसकी विशेषताएँ क्या हैं?",                "hindi"),
    ("मेरे दादा की सेहत बहुत खराब हो गई है, मुझे क्या करना चाहिए?",        "hindi"),
    ("Ek chhota sa kahani likho jismein ek jadugar aur ek rajkumar ho.", "hinglish"),
]

import os, re, json, time, shutil, urllib.request, urllib.error, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import contextlib
@contextlib.contextmanager
def _null(): yield
torch.no_grad = lambda f=None: (_null() if f is None else (lambda g: g))
torch.cuda.OutOfMemoryError = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", RuntimeError)

os.chdir("/tmp")
SMALL = ["config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
         "vocab.json","merges.txt","added_tokens.json","special_tokens_map.json"]

def fetch_repo(repo, dest):
    def get(fn):
        tok = os.environ.get("HF_TOKEN","").strip()
        h = {"Authorization": f"Bearer {tok}"} if tok else {}
        with urllib.request.urlopen(urllib.request.Request(
                f"https://huggingface.co/{repo}/resolve/main/{fn}", headers=h), timeout=600) as r:
            return r.read()
    os.makedirs(dest, exist_ok=True)
    have = set(os.listdir(dest)); idx = "model.safetensors.index.json"
    need = list(SMALL)
    if idx not in have:
        try:
            open(os.path.join(dest,idx),"wb").write(get(idx)); have.add(idx)
            need += sorted(set(json.load(open(os.path.join(dest,idx)))["weight_map"].values()))
        except Exception:
            need.append("model.safetensors")
    tot = 0; t0 = time.time()
    for fn in need:
        dst = os.path.join(dest, fn)
        if fn in have and os.path.getsize(dst) > 0: continue
        try: body = get(fn)
        except urllib.error.HTTPError as e:
            print(f"    skip {fn} ({e.code})"); continue
        open(dst,"wb").write(body); tot += len(body)
        print(f"    {fn:30s} {len(body):>12,}", flush=True)
    print(f"    fetched {tot/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)
    return dest

DEV_SYS = ("Aap ek helpful assistant ho. User ke sawaal ka jawab POORI tarah se, saaf "
           "aur natural Devanagari Hindi mein likho. English se seedha translation mat karo. "
           "Jawab 30 se 50 shabd ka likho.")
HING_SYS = ("Aap ek helpful assistant ho. User Hinglish mein likhta hai. Aap bhi romanized "
            "Hinglish mein jawab do. Poora aur sahi jawab do, 30 se 50 shabd mein.")
ENDS = re.compile(r"[।.?!\"'”’)\]]\s*$")

rows = []
for repo in CANDIDATES:
    print(f"\n{'='*70}\n{repo}", flush=True)
    dest = f"/tmp/t_{repo.split('/')[-1]}"
    t_dl = time.time()
    fetch_repo(repo, dest)
    dl = time.time() - t_dl
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tk = AutoTokenizer.from_pretrained(dest)
    if tk.pad_token_id is None: tk.pad_token = tk.eos_token
    md = AutoModelForCausalLM.from_pretrained(dest, quantization_config=bnb, device_map="auto").eval()
    tk.padding_side = "left"
    tot_t = 0.0; n_ok = 0; words = []; devs = []
    for prompt, lang in PROBE_PROMPTS:
        sysp = DEV_SYS if lang == "hindi" else HING_SYS
        txt = tk.apply_chat_template([{"role":"system","content":sysp},
                                      {"role":"user","content":prompt}],
                                     add_generation_prompt=True, enable_thinking=False, tokenize=False)
        enc = tk([txt], return_tensors="pt").to(md.device)
        t0 = time.time()
        with torch.no_grad():
            o = md.generate(**enc, max_new_tokens=380, do_sample=False,
                            pad_token_id=tk.pad_token_id)
        dt = time.time() - t0; tot_t += dt
        a = tk.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        w = len(a.split()); d = sum("ऀ"<=c<="ॿ" for c in a)/max(1,len(a))
        complete = bool(ENDS.search(a))
        words.append(w); devs.append(d); n_ok += int(complete and 30 <= w <= 160)
        rows.append({"repo":repo,"lang":lang,"prompt":prompt,"reply":a,
                     "words":w,"dev":round(d,3),"complete":complete,"sec":round(dt,1)})
        print(f"  [{lang:7s}] {w:3d}w  dev {d:.2f}  {'OK ' if complete else 'TRUNC'}  {dt:5.1f}s", flush=True)
    med = sorted(words)[len(words)//2]
    print(f"\n  -> complete+in-range {n_ok}/{len(PROBE_PROMPTS)}   median {med}w   "
          f"{tot_t/len(PROBE_PROMPTS):.1f}s/answer   download {dl:.0f}s", flush=True)
    print(f"     SECOND GPU IDLE? device_map='auto' fits 4-bit in 15GB -> gpu0 only. "
          f"Run a 2nd process with CUDA_VISIBLE_DEVICES=1 for a free 2x.", flush=True)
    del md; torch.cuda.empty_cache()

print(f"\n{'='*70}\nSUMMARY  (higher better except sec)\n{'='*70}")
print(f"{'model':32s} {'ok':>5s} {'medW':>5s} {'medDev':>7s} {'s/ans':>7s}")
for repo in CANDIDATES:
    r = [x for x in rows if x["repo"]==repo]
    if not r: continue
    ok = sum(x["complete"] and 30<=x["words"]<=160 for x in r)
    med = sorted(x["words"] for x in r)[len(r)//2]
    md_ = sorted(x["dev"] for x in r)[len(r)//2]
    st = sum(x["sec"] for x in r)/len(r)
    print(f"{repo:32s} {ok}/{len(r):<3d} {med:>5d} {md_:>7.2f} {st:>7.1f}")
json.dump(rows, open("/tmp/teacher_probe.json","w"), ensure_ascii=False, indent=1)
print("\nsamples written to /tmp/teacher_probe.json - read them before trusting the table")
