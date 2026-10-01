#!/usr/bin/env python3
"""Grow the prompt pool. 20 questions per call for the price of one answer, so
this is the cheapest way to add volume: ~600 calls is a couple of minutes."""
import json, re, os, time, random, threading, urllib.request
import concurrent.futures as cf
random.seed(41)

CFG = "/home/ubuntu/.pi/agent/models.json"
_p = json.load(open(CFG))["providers"]["openrouter"]
BASE, KEY, MODEL = _p["baseUrl"], _p["apiKey"], "stealth/space-bunny-alpha"
OUT = "/home/ubuntu/hindi-finetune/teach/extra_prompts.jsonl"
DEV = lambda s: sum(1 for c in s if "ऀ" <= c <= "ॿ") / max(1, len(s))

SUBJ = ["daily routine","health and fitness","cooking at home","travel and trains","study and exams",
 "job and interviews","relationships","money and saving","technology and phones","books and reading",
 "cinema and music","sports","village life and farming","weather and seasons","animals and pets",
 "gardening","friendship","shopping and clothes","vehicles and traffic","government schemes",
 "education system in India","history and freedom struggle","science and space","doctors and medicines",
 "computer problems and software","small businesses","art and painting","photography",
 "home repair and electricity","water and sanitation","children and school","senior citizens",
 "women and society","environment and pollution","festivals","food habits","memory and concentration",
 "news and newspapers","social media","online shopping","food delivery","banks and loans",
 "insurance and paperwork","government exams","resumes and interviews","startups","shops and retail",
 "farming and crops","rain and rivers","forests and trees","mountains and travel","beaches",
 "city life","traffic and crowds","recycling and waste","radio and television","mobile apps",
 "electricity bills","internet problems","clothing and tailoring","skin and beauty","hair problems",
 "weight and fitness","stress and mental health","sleep and tiredness","nutrition","tea and coffee",
 "markets and bargaining","arguments with friends","family and relatives","marriage and weddings",
 "pregnancy and children","cold and cough","fever","injuries and first aid","medicines",
 "police and legal help","courts and lawyers","tax and income","parcels and courier",
 "queueing and waiting","trains and buses","second hand shopping","laptops and computers",
 "mobile repairs","internet plans","electricity and power cuts","water problems","cooking oil and gas",
 "rice and wheat","milk and dairy","tea gardens","cattle and livestock","poultry and eggs",
 "handicrafts","folk dance","classical music","cricket and football","badminton and fitness",
 "swimming","cycling","mountaineering","camping","picnic","weddings and rituals","funeral customs",
 "diwali and holi","eid and christmas","greetings and wishes","gift giving","guest hospitality"]
ANGLE = ["a beginner asking for help", "someone confused by rules and paperwork",
         "a person comparing two options", "someone describing a problem and asking what to do",
         "a person asking why something is the way it is", "someone asking for a short list or steps"]
SYS = ("Aap ek data assistant ho. Sirf list format mein jawab do, koi explanation nahi. "
       "20 alag aur alag sawaal likho.")

def ask(topic, angle, lang):
    if lang == "hinglish":
        req = (f"Write 20 short questions a real user would type in HINGLISH (romanized Hindi "
               f"mixed with English, exactly how an Indian types on a phone). Topic: {topic}. "
               f"Angle: {angle}. Rules: each 5 to 15 words; natural and varied; each question "
               f"completely self-contained with no reference to anything said earlier; NO numbering, "
               f"NO bullets, NO answers; one per line.")
    else:
        req = (f"20 प्रश्न लिखो जो कोई सच्चा व्यक्ति {topic} के बारे में पूछेगा। परिप्रेक्ष्य: {angle}। "
               f"नियम: हर प्रश्न 5 से 15 शब्द का; विविध; हर प्रश्न पूरी तरह समझ आने वाला हो, पहले किसी "
               f"बात का ज़िक्र न हो; नंबरिंग नहीं, बुलेट नहीं, उत्तर नहीं; हर प्रश्न अलग लाइन में।")
    b = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": SYS}, {"role": "user", "content": req}],
        "temperature": 1.0, "max_tokens": 2000}).encode()
    r = urllib.request.Request(BASE + "/chat/completions", data=b, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(r, timeout=180) as x:
                return json.loads(x.read())["choices"][0]["message"]["content"] or ""
        except Exception:
            time.sleep(2 * (a + 1))
    return ""

NUM = re.compile(r"^\s*(\d+[\.\)]|[-*•])\s*")
lock = threading.Lock(); rows = []
def work(t):
    topic, angle, lang = t
    got = []
    for l in ask(topic, angle, lang).split("\n"):
        l = NUM.sub("", re.sub(r"^[\s\"'#*-]+", "", l)).strip().strip('"')
        if not (4 <= len(l.split()) <= 40) or len(l) < 10: continue
        d = DEV(l)
        if lang == "hinglish" and d > 0.35: continue
        if lang == "hi" and d <= 0.65:      continue
        got.append((l, lang))
    with lock: rows.extend(got[:20])
    return len(got[:20])

jobs = [(t, a, "hi") for t in SUBJ for a in ANGLE]
jobs += [(t, a, "hinglish") for t in SUBJ for a in ANGLE]
random.shuffle(jobs)
print(f"{len(jobs)} calls ({len(SUBJ)} topics x {len(ANGLE)} angles x 2 languages)", flush=True)
t0 = time.time(); tot = 0
with cf.ThreadPoolExecutor(max_workers=24) as ex:
    for i, n in enumerate(ex.map(work, jobs), 1):
        tot += n
        if i % 100 == 0: print(f"  {i}/{len(jobs)} calls  {tot:,} prompts  {time.time()-t0:.0f}s", flush=True)
seen = set(); uniq = []
for t, lang in rows:
    k = t.lower()
    if k in seen: continue
    seen.add(k); uniq.append({"instruction": t, "plang": lang, "src": "expand"})
with open(OUT, "w", encoding="utf-8") as f:
    for r in uniq: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nwrote {len(uniq):,} unique new prompts -> {OUT}   ({time.time()-t0:.0f}s)")
