#!/usr/bin/env python3
"""Generate a balanced prompt bank with the teacher itself.

The local pool had 948 Hinglish prompts against 26,131 Hindi/English ones, so a
random shuffle produced 1% Hinglish. Asking the teacher for the prompts is far
cheaper than answering them: one call returns 20 questions for the price of one
answer, and it lands in the register we actually want the answers to be in.
"""
import json, re, os, random, time, threading, urllib.request
import concurrent.futures as cf
random.seed(5)

CFG = "/home/ubuntu/.pi/agent/models.json"
OUT = "/home/ubuntu/hindi-finetune/teach/boot_prompts.jsonl"
_p = json.load(open(CFG))["providers"]["inceptionlabs"]
BASE, KEY, MODEL = _p["baseUrl"], _p["apiKey"], "mercury-2.5"

TOPICS = """daily routine and habits|health and fitness|cooking at home|travel and trains|
study and exams|job and interviews|relationships and dating|money and saving|
technology and phones|books and reading|cinema and music|sports and cricket|
village life and farming|weather and seasons|animals and pets|gardening|
friendship and neighbours|shopping and clothes|vehicles and traffic|
government schemes and documents|education system in India|history and freedom struggle|
science and space|health problems and doctors|computer problems and software|
business and small shops|business and startups|art and painting|photography|
home repair and electricity|water and sanitation|children and school|
senior citizens and parents|women and society|environment and pollution|
festivals and celebrations|food habits and eating|memory and concentration|
"""
TOPICS = [t.strip() for t in TOPICS.replace("\n","").split("|") if t.strip()]

SYS = ("Aap ek data assistant ho. Aap sirf list ke format mein jawab dete hain, koi "
       "explanation nahi. Hamesha 20 alag aur alag sawaal likho.")

def ask(topic, lang):
    if lang == "hinglish":
        req = (f"Write 20 short questions a real user might type in HINGLISH "
               f"(romanized Hindi mixed with English, written exactly how an Indian "
               f"person types on a phone) about: {topic}. Rules: each question 5 to 15 "
               f"words; natural and varied; some factual, some asking for advice, some "
               f"personal; NO numbering, NO bullets, NO answers; one question per line.")
    else:
        req = (f"20{lang}20 वे प्रश्न लिखो जो कोई सच्चा व्यक्ति {topic} के बारे में पूछेगा। "
               f"नियम: हर प्रश्न 5 से 15 शब्द का; सामान्य और विविध; कुछ तथ्यात्मक हों, कुछ "
               f"सलाह माँगने वाले हों; नंबरिंग नहीं, बुलेट नहीं, उत्तर नहीं; हर प्रश्न अलग लाइन में।")
    body = json.dumps({"model": MODEL, "messages": [
        {"role":"system","content":SYS},{"role":"user","content":req}],
        "temperature":1.0, "max_tokens":4000}).encode()
    r = urllib.request.Request(BASE+"/chat/completions", data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type":"application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(r, timeout=180) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(2*(a+1))
    return ""

DEV=lambda s: sum(1 for c in s if "ऀ"<=c<="ॿ")/max(1,len(s))
NUM=re.compile(r"^\s*(\d+[\.\)]|[-*•])\s*")
out, lock = [], threading.Lock()

def work(t):
    r = t
    lines = [re.sub(r"^[\s\"'#*-]+","",l).strip() for l in r.split("\n") if l.strip()]
    rows=[]
    for l in lines:
        l = NUM.sub("", l).strip().strip('"')
        if not (4 <= len(l.split()) <= 40): continue
        if DEV(l) > 0.35: continue          # Hinglish bank must be romanized
        if len(l) < 8: continue
        rows.append(l)
    with lock:
        out.extend(rows[:20])
    return len(rows[:20])

jobs = [(t,"hinglish") for t in TOPICS] + [(t,"hi") for t in TOPICS]
t0=time.time()
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    got = list(ex.map(lambda a: work(ask(*a)), jobs))
print(f"{len(TOPICS)} topics x 2 languages, {sum(got)} prompts in {time.time()-t0:.0f}s")
seen=set(); uniq=[]
for p in out:
    k=p.lower()
    if k in seen: continue
    seen.add(k); uniq.append(p)
with open(OUT,"w",encoding="utf-8") as f:
    for p in uniq:
        f.write(json.dumps({"instruction":p,"plang":"hinglish","src":"boot"},ensure_ascii=False)+"\n")
print(f"wrote {len(uniq):,} unique bootstrapped prompts -> {OUT}")
print("samples:")
for p in uniq[:6]: print(f"   - {p}")
