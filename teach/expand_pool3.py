#!/usr/bin/env python3
"""Second prompt-pool expansion. 111 subjects produced 12,545 distinct Devanagari
forms - 8.4% of a general dictionary. Vocabulary breadth is a function of subject
breadth, so this widens SUBJ from 111 to ~500, weighted toward the domains the
first pass never touched: science, medicine, law, engineering, finance, geography.
Those are also where the existing corpus is worst - 2.1% of technical vocabulary.
"""
import json, random, re, threading, urllib.request, queue, os
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_cfg import TEACHER   # baseUrl/apiKey/model from gitignored config

OUT = "/home/ubuntu/hindi-finetune/teach/extra_prompts3.jsonl"
MODEL = TEACHER["model"]
LIMIT = int(os.environ.get("LIMIT", "5000"))

# SIX questions per call, not one. At one-per-call the pool grew 616 prompts from
# 3,336 calls - a 16% keep rate that made the pool, not the generator, the ceiling
# on corpus size. Six per call turns the same 3,336 calls into ~20,000 candidates.
SYS = ("You write realistic user questions that a Hindi or Hinglish speaker would "
       "actually type to an AI assistant. Write SIX different questions, one per "
       "line, each on its own line, no numbering, no bullets, no quotes, no "
       "explanation. Each question under 25 words. Vary the wording and intent.")

# The apiKey lives at PROVIDER level in models.json, not on the model entry -
# reading it off the model dict is why the first launch died with KeyError.
BASE, K = TEACHER["baseUrl"], TEACHER["apiKey"]
MODEL = TEACHER["model"]
LOCK = threading.Lock(); Q = queue.Queue(); DONE = [0]; KEPT = []
STOP = threading.Event()

def call(user):
    body = json.dumps({"model": MODEL, "messages": [{"role": "system", "content": SYS},
                                                    {"role": "user", "content": user}],
                       "temperature": 0.9, "max_tokens": 700}).encode()
    r = urllib.request.Request(BASE + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def norm(t, lang):
    t = t.strip().strip('"').strip()
    t = re.sub(r"^[-*\u2022]?\s*\d+[.)]\s*", "", t).strip().strip('"')
    if not (10 <= len(t.split()) <= 28): return None
    if re.search(r"(https?://|\d{3,}|lorem|TODO)", t): return None
    dev = sum(1 for c in t if "ऀ" <= c <= "ॿ") / max(1, len(t))
    if lang == "hi" and dev < 0.55: return None
    if lang == "hinglish" and dev > 0.35: return None
    if not t.endswith(("?", "?", "?", "।", ".", "!")): return None
    return {"instruction": t, "plang": lang, "src": "teacher_pool_v2"}

def work():
    while not STOP.is_set():
        try: subj, angle, lang = Q.get_nowait()
        except queue.Empty: return
        try:
            raw = call(f"Write SIX questions in {'Hindi (Devanagari)' if lang=='hi' else 'Hinglish, written in Latin script as Indians type it'} "
                       f"about \"{subj}\" from the point of view of {angle}. "
                       f"Six different questions, one per line, no numbering.")
            got = []
            for line in raw.split("\n"):
                p = norm(line, lang)
                if p: got.append(p)
            if got:
                with LOCK: KEPT.extend(got)
        except Exception: pass
        finally:
            with LOCK: DONE[0] += 1

SUBJ = [
 # science and engineering - worst covered, added heavily
 "physics and mechanics","chemistry at school level","electric circuits and wiring","thermodynamics and heat",
 "astronomy and telescopes","genetics and heredity","microbiology and germs","botany and plant growth",
 "geology and earthquakes","oceanography and tides","meteorology and clouds","ecology and pollution",
 "robotics and automation","machine learning basics","cybersecurity and passwords","networking and Wi-Fi",
 "electronics and circuit boards","3D printing","renewable energy and solar","nuclear power",
 # medicine and health
 "dentistry and dental care","pregnancy and childbirth","mental health and anxiety","diet and nutrition",
 "physiotherapy and exercise injuries","vaccination and immunity","allergies and skin problems","sleep disorders",
 "eye health and glasses","hearing and ear problems","orthopedics and bone health","oncology and cancer care",
 "Ayurveda and home remedies","first aid and emergencies","epidemics and public health",
 # law and administration
 "land laws and property papers","rental agreements and tenants","labour law and workers' rights",
 "court procedures and legal notices","police procedures and FIR","tax filing and GST",
 "immigration and passports","election law and voting rights","consumer protection and warranties",
 "pension schemes and retirement benefits",
 # finance and commerce
 "accounting and bookkeeping","stock markets and mutual funds","insurance claims","loans and EMI planning",
 "cryptocurrency basics","small business registration","real estate and property valuation","exports and imports",
 "advertising and marketing","ecommerce and online selling",
 # engineering and industry
 "civil engineering and construction","mechanical engineering basics","electrical wiring and safety",
 "automobile repair and maintenance","aviation and air travel rules","shipbuilding and ports",
 "mining and minerals","textile and garment industry","food processing and FSSAI","chemical safety in industry",
 # geography and environment
 "monsoon and rainfall patterns","rivers and dams","deserts and drought","coastal and marine life",
 "glaciers and climate change","forests and wildlife","national parks and sanctuaries","soil and agriculture",
 # history and culture
 "Mughal history","British colonial period in India","freedom movement and independence",
 "ancient Indus Valley Civilisation","medieval Indian kingdoms","Maratha empire","Bhakti and Sufi movements",
 "Indian classical music","Indian classical dance","Hindu festivals and their meaning","Islamic festivals in India",
 "regional folk traditions","Indian painting styles","Indian cinema history","literature and poetry",
 # education and exams
 "engineering entrance exams","medical entrance exams","competitive examinations","board examinations",
 "scholarships and education loans","online education and degrees","language learning and grammar",
 # work and business
 "startups and business plans","freelancing and remote work","government jobs and examinations",
 "corporate workplaces and HR","sales and negotiation","restaurant and hotel business",
 "farming and dairy business","handicrafts and small manufacturing",
 # everyday practical
 "electricity bills and power cuts","water supply and tanker water","gas cylinders and cooking gas",
 "mobile recharge and plans","passport and travel documents","vehicle insurance claims","traffic rules and fines",
 "waste disposal and recycling","pension planning after retirement","children's schooling and homework",
 # --- second pass: breadth beyond the first 113 ---
 "acid rain and air quality","volcanoes and earthquakes","acoustics and sound waves","lenses and optics",
 "radioactivity and X-rays","plasma and neon lights","batteries and charging","sensors and IoT devices",
 "satellite communication","weather balloons and forecasting","glaciology and ice cores","soil fertility and compost",
 "beekeeping and honey","sericulture and silk","fisheries and fish farming","horticulture and pruning",
 "floriculture and flower arrangements","tea and coffee cultivation","spice farming and processing","cold storage and supply chains",
 "warehouses and inventory","packaging and labelling","quality control in factories","factory safety rules",
 "fire safety and extinguishers","first aid kits at home","ambulance and emergency numbers","blood donation and transfusion",
 "organ donation","clinical trials and approvals","generic medicines and patents","pharmacy practice and prescriptions",
 "telemedicine and online consultation","health insurance schemes","Ayurvedic and Unani medicine comparison",
 "dieting and intermittent fasting","sports injuries and recovery","yoga and breathing exercises",
 "running and marathon training","swimming and water safety","cycling and road safety","martial arts and self defence",
 "chess and board games","video games and screen time","photography and exposure","film making and editing",
 "animation and VFX","graphic design and typography","calligraphy and lettering","pottery and ceramics",
 "woodworking and carpentry","weaving and handloom","embroidery and needlework","jewellery making and silver smithing",
 "printing and publishing","newspaper and journalism","radio and podcasting","social media and content creation",
 "digital marketing and SEO","e-commerce logistics and returns","cryptocurrency and blockchain basics",
 "personal finance and budgeting","credit score and loans","bequests and wills","insurance for small shops",
 "co-operative societies","SHGs and microfinance","crop insurance and MSP","land records and khatauni",
 "boundary disputes and survey","building permissions and plan sanction","rainwater harvesting",
 "septic tanks and groundwater","waste picking and recycling economy","plastic ban and alternatives",
 "wildlife corridors and human conflict","mangroves and coastal erosion","desertification and afforestation",
 "rivers of India and their tributaries","Himalayan geography and glaciers","Deccan plateau and rivers",
 "islands of India and tourism","border areas and security","tribal communities and welfare",
 "caste and social justice in India","gender and women's rights","child labour laws and education",
 "disability rights and accessibility","senior citizen schemes and pensions","widow pension schemes",
 "Aadhaar and digital identity","DigiLocker and government services","UPI and digital payments",
 "banking correspondents in villages","co-operative bank failures","microfinance and repayment stress",
 "MGNREGA and rural work","PMAY and housing schemes","Swachh Bharat and sanitation","Ayushman Bharat health cover",
 "PM-KISAN and farm income","START India and small business loans","patent filing and intellectual property",
 "startup incubators and funding","angel investors and pitch decks","failure and pivot in business",
 "time management and procrastination","memory and revision techniques","note making and summarising",
 "speed reading and comprehension","public speaking and stage fear","interview preparation and body language",
 "workplace communication and emails","teamwork and delegation","leadership and feedback",
 "negotiation and persuasion","conflict resolution at work","remote work and timezone coordination",
 "freelance platforms and rates","portfolio and personal branding","burnout and rest",
 "attachment and parenting","discipline and positive reinforcement","teenagers and screen addiction",
 "single parent households","elderly care and assisted living","marriage and inter-caste families",
 "friendship and boundaries","loneliness in cities","homesickness and migration",
 "addiction and recovery","anger and stress management","grief and loss","meditation and mindfulness",
 "astrology and numerology beliefs","superstition and science","religion and personal faith","ethics and moral dilemmas",
 "corruption and civic behaviour","bureaucracy and paperwork delays","right to information and RTI",
 "public protests and civic movements","voting and political awareness","news media and fact checking",
 "fake news and misinformation","social media regulation","digital literacy and privacy",
 "online scams and fraud","cyberbullying and harassment","screen time and eyesight",
 "coaching centres and tuition","competitive exam preparation strategy","interview and group discussion",
 "engineering and medical entrance coaching","skill development and vocational training","apprenticeship schemes",
 "textile and apparel exports","IT and software services industry","BPO and outsourcing sector",
 "tourism and hospitality sector","film and entertainment industry","sports and athletics industry",
 "banking and financial services sector","agriculture and food processing sector","renewable energy sector",
]

for s in SUBJ: s = s.strip()
SEEN, UNIQ = set(), []
for s in SUBJ:
    if s and s not in SEEN: SEEN.add(s); UNIQ.append(s)
print(f"{len(UNIQ)} subjects x 6 angles x 2 languages = {len(UNIQ)*12} prompt calls", flush=True)

JOBS = [(s, a, l) for s in UNIQ for l in ("hi", "hinglish")
        for a in ("a beginner asking for help", "someone confused by rules and paperwork",
                  "a person comparing two options", "someone describing a problem and asking what to do",
                  "a person asking why something is the way it is", "someone asking for a short list or steps")]
random.Random(7).shuffle(JOBS)
for j in JOBS[:LIMIT]:
    Q.put(j)

TH = [threading.Thread(target=work, daemon=True) for _ in range(16)]
for t in TH: t.start()
try:
    while any(t.is_alive() for t in TH):
        pass
        import time as _t; _t.sleep(20)
        with LOCK:
            print(f"  {DONE[0]:>5} calls  kept {len(KEPT):>5}", flush=True)
except KeyboardInterrupt:
    STOP.set()
with LOCK:
    with open(OUT, "w", encoding="utf-8") as f:
        for p in KEPT: f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"wrote {len(KEPT):,} prompts -> {OUT}")
