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

OUT = "/home/ubuntu/hindi-finetune/teach/extra_prompts5.jsonl"
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
 # --- round 3: 600+ more subjects, Hinglish-weighted ---
 "bimaru labh aur chhoti naukri","kisan ke bhaag daar","doodh aur khane ka bazaar","sone ki sona",
 "ghar banana aur kirayedar","majdoore aur mizan","kisan ka bima","kutte aur billi ki dekhbhal",
 "baap ghar aur sasural kaatna","saas bahu ke taluq","ladaai aur manaana","padhai ke Charges",
 "phone ka battery jaldi khatam","net ki speed slow","TV par koi nahi dikha raha","password bhool gaye",
 "UPI se paisa nahi gaya","bank account band ho gaya","Aadhaar se naam theek nahi hua","PAN card kaise banaye",
 "ghar ki bijli ka bill badh gaya","paani nahi aa raha","gas cylinder nahi mila","ration card ka kaam",
 "MGNREGA kaam nahi mila","PMAY ghar nahi mila","hosipal me bheed lagi","doctor ne dawa nahi di",
 "jab thand lagti hai","jab bukhar aata hai","bachche ko bukhar aaya","budhe ko chakkar aata hai",
 "hamare masalay ki shararat","bacche school se nahi jaate","padhai mein man nahi lagta",
 "exam ki taiyari kaise kare","interview mein kya bole","resume kaise banaye","job dhundne ka tareeka",
 "office mein boss se complaint","salary nahi mili","PF aur EPF ka fayda","notice period kya hota hai",
 "matlab aur nirman ka antar","akṣar vs akshar","gyan kaise badhaye","kitab parhay ki kare",
 "sher kehte hain ya sher","do-doosre ki shadi","hum aap kaise hain","main aapse mohabbat karta hoon",
 "gusse par control","chinta kaise kam kare","neend nahi aati","dimag mein bahut saare vichar",
 "phone par ghanton","social media addiction","online paise kamane ka tareeka","crypto samajh nahi aata",
 "AI se kaam karwana","photo banana seekhna","video edit karna","blog likhna",
 "khana banana nahi aata","subzi banane ka nuskha","chai ki recipe","Roti kaise banaye",
 "machhli kaise pakode","salad ka recipe","diet plan","roti ke kadhi mein kya daale",
 "ghar ki saaf safai","bathroom cleaning tips","kapde dhone ka tareeka","iron karne ka sahi tareeka",
 "bheed se kaise bach","sadke ka rasta","bhus aur barish","currence of traffic",
 "bus ka schedule","train ka ticket book kare","flight me bag check kare","hotel ka room book kare",
 "visa lagwane ka process","passport banwane ki documents","police station jaana hai","FIR kaise likhe",
 "court me kistab de","vakil se baat kare","bail milega kya","RTI file karna hai",
 "voter ID banwana","election me vote dena","political party kaise chune","sarkar ki yojana",
 "subsidy kaise mile","bank loan ke documents","EMI kaise kam kare","savings kaise kare",
 "tax file karna hai","GST number lena hai","FIR se kaise bach","khud ka kanoon",
 "doodh ka adarc","aam ka carkar","haldi ka teer","baraf ke baare me","bijli ke kharch",
 "ghaas katne ke upay","kuaan gehra karna","daim kya hota hai","baarish ka nuksan",
 "bacta kya hai","bacteria ke baare me","kya kya bimariyan hoti hain","immun kya hai",
 "allergy se ghabrahat","BP high hone ke karan","sugar ki bimari","BP medicine","thyroid kya hota hai",
 "migrane ka dard","juri se kamal","budhapa kya hota hai","life insurance lena chahiye",
 "nirjeevan bima","Swachh Bharat","sanitation ki samasya","recycling kaise kare",
 "kudala kahan fenke","plastic ki problem","jalvayu parivartan","vayu pradushan",
 "sunami se kaise bache","bhukamp kya hota hai","chhata kya hai","garam garmi kya hai",
 "Stark ka arth","quantum computer","AI ka future","robot kaise banate hain",
 "satellite kaise chalta hai","rocket science","nuclear power plant","solar panel kaise lagta hai",
 "battery kaise banate hain","mobile ka circuit","computer ka RAM","hard disk kya hai",
 "internet kaise chalta hai","firewall kya hai","virus se kaise bachaye","data kya hota hai",
 "coding seekhna","Python kya hai","website kaise banaye","app kaise banaye",
 "game kaise banaye","3D kya hota hai","animation kaise banti hai","film shoot kaise hota hai",
 "camera kaise chalate hain","mic kaise lagate hain","stage ka performance","acting kya hoti hai",
 "script writing","play likhna","dance kaise seekhe","sing kaise gaaen",
 "guitar kaise baaje","piano sikhna","painting kaise kare","sketch kaise banaye",
 "craft kaise banaye","cooking channel banana","blog kaise likhe","news kaise banate hain",
 "newspaper mein kaise kaam kare","reporter kaise bane","editor kya hota hai",
 "anchor kaise bane","YouTube channel kaise chalaye","blogger kaise bane",
 "elon musk kya hai","modi ji kaun hain","gandhi ji kaun the","ambedkar kaun the",
 "shakespeare kaun tha","newton kaun the","einstein kaun the","tagore kaun the",
 "premier league kya hai","ipl kya hai","cricket ka rule","football ka rule",
 "badminton kaise khele","khilone kya kya hai","shakkar kya hai","tir ke khel",
 "fitness kaise kare","gym kaise jaye","yoga ka fayda","running se nuksan",
 "diet kaise kare","BP me kya kare","neend ki dawai","mothi ka ilaj",
 "neti kya hai","bhi pili","doodh ke fayde","lehi ka fayda","nimbu paani",
 "nimbu sherbet","faluda","upma","dosa","idli","vada","chole bhature",
 "paratha","khichdi","dal roti","sabzi","achaar","papad","pakora",
 "mithai ka nam","barfi","laddu","jalebi","rasgulla","kulfi",
 "chai ki kahani","tapri","paan shop","chai patti","adrak wali chai",
 "gannas juice","sattu","bajra roti","jowar roti","makki ki roti",
 "bajra ka atta","peanut","sarson ka tel","til ka tel","sarso ka sarson",
 "cotton","kapas","kisan fasal","gehu","chawal","dal","ganna",
 "kisan kitna kamata","mandi bhav","kisan ki ksht","rajasthan ki kheti",
 "punjab ki kheti","maharashtra ke kisan","kashmir ka khajoor","assam ka chai",
 "darjeeling ki chai","coorg ki coffee","goa ki mawa","kerala ki nariyal",
 "bengal ka rasgulla","tamil film","bollywood","kollywood","tollywood",
 "hindustani music","carnatic music","bhangra","kathakali","odissi",
 "bihar ka dance","gujarati naat","punjabi giddha","kashmiri wazwan",
 "taj mahal","red fort","india gate","gateway of india","ganga",
 "yamuna","ganga river","bharatpur","hawa mahal","golden temple",
 "meenakshi temple","jagannath temple","kerala backwaters","himalayas",
 "ladakh","goa beach","andaman","darjeeling hill","ooty","kerala monsoon",
 "kashmir valley","ladakh winter","chennai traffic","mumbai local",
 "delhi metro","bangalore it hub","hyderabad pharma","pune software",
 "ahmedabad textile","surat diamond","coimbatore cotton","kanpur leather",
 "moradabad brass","firozabad glass","jodhpur marble","kullu shawl",
 "chamba rumal","kashmiri carpet","tussar silk","banarasi sari",
 "kanjeevaram","chikankari","phulkari","zardozi","chikankari ka kaam",
 # --- round 4 ---
 "NEP 2020 kya hai","NEP ke fayde","board exam me 90% kaise laaye","NEET ki taiyari",
 "JEE Main kya hota hai","CUET ka kya hai","NDA kya hai","SSC CGL taiyari",
 "Railway ka exam","banking ka exam","CAT kya hai","GATE kya hai",
 "CSIR NET","UPSC prelims","state PSC","CDS exam"," NDA exam",
 "chattrapati Shivaji Maharaj","Rani Lakshmibai","Bhagat Singh","Sardar Patel",
 "Jawaharlal Nehru","Subhash Chandra Bose","Mahatma Gandhi ke updesh","satyagrah kya tha",
 "non-cooperation movement","civil disobedience","Dandi march","quit India movement",
 "Jallianwala Bagh","Bhangra ke gaanth","Champaran satyagraha","Kheda satyagraha",
 " Bardoli satyagraha","indian independence ke praroopak","Pehli war","1857 ka Bidroha",
 "Sipahi mutiny","Mauj","Rani Lakshmibai ka role","Tana Bhagat",
 "Bhagat singh ke prabhav","ghadar party","Lala Lajpat Rai","Bhikaji Cama",
 "Annie Besant","Jamnalal Bajaj","Hakim Ajmal Khan","Abul Kalam Azad",
 "Maulana Abul Kalam Azad","Maulana Azad ka jeevan","Netaji ka INLA","Azad Hind Fauj",
 "Rani Durgavati","Chhatrapati Shivaji ka swaroop","Shivaji ki olkhanensi","Mavlas",
 "tanaji Malusare","Afzal Khan","TANODAR","Hindavi Swaraj","Peshwa Baji Rao",
 "Maratha confederacy","Shivaji ki navy","Hindavi language","Nautical training",
 "ancient India ka itihas","Indus valley civilization","Harappa","Mohenjo-daro",
 "vedas ka gyan","Rigveda","Yajurveda","Samaveda","Atharvaveda",
 "Upanishad","Panchatantra ka kahani","Jataka story","Buddh ki katha",
 "Mahabharat ka gyan","Ramayan ki kahani","Krishn","Arjun","Bhishma",
 "Gandhari","Karna","Abhimanyu","Ashoka ka chakra","Ashoka ka dhamma",
 "Chandragupta Maurya","Ashoka the great","Gupta dynasty","Vikramaditya",
 "Chandrakeertika","Harsha Vardhana","Prithviraj Chauhan","Muhammad bin Qasim",
 "Mughal badshah","Akbar","Shah Jahan","Aurangzeb","Jahangir",
 "Humayun","Babur","Sher Shah Suri","Sardar Khan","Rani Rudrama Devi",
 "Krishnadevaraya","Shivaji ka shastra","Kautilya ka arthashastra","Chanakya ka nitisastra",
 "Kashmir ka itihas","Kashmir me badal","Kashmir se related kavita","Dogra dynasty",
 "Nepal ka itihas","Nepal me tourism","Bhutan ka rahasya","Tibet ka sanskrit",
 "Sikkim ka kahawa","Assam ka ahom","Ahom dynasty","Manipur ka Manipuri",
 "Manipuri dance","Naga culture","Nagaland ke warrior","Tribe ki traditions",
 "Adivasi Sanskriti","sanctuary me rehti jaanwarian","tiger ka baccha","Project Tiger",
 "one horned rhinoceros","snow leopard","elephant sanctuary","bird sanctuary",
 "wetland of india","Sundarbans","Ranthambore","Kanha National Park","Bandipur",
 "Kaziranga","Periyar","Gir National Park","Rajasthan ka desert","Thar desert",
 "Rann of Kutch","Gujarat ke forests","Mizoram ka bamboo","Manipur ke hills",
 "Kashmir ka saffron","Kashmiri apple","Saffron kaise ugta hai","apple ki kheti",
 "grapes ki kheti","banana ki kheti","sugarcane ka udhyog","sugar mill",
 "Rice mill","textile mill kaise chalta hai","handloom ka kaam","silk ka udhyog",
 "jute ka udhyog","wool industry","tea garden","coffee plantation",
 "spice export","onion ki mandi","tomato ka bhav","potato ki kheti",
 "sabzi mandi","cold storage","kisan e-nam","PM Kisan se paise kab aate hain",
 "Kisan Credit Card","Kisan ki fasal ka bima","PM Fasal Bima Yojana",
 "soil health card","Kisan Call Centre","Krishi Vigyan Kendra","agriculture university",
 "animal husbandry","poultry farming","goat farming","fish farming",
 "apiculture","beekeeping ki kahani","dairy farm","Amul ka bana hua","dairy cooperative",
 "aata kaise banta hai","atta mill","sugar mill kaise chalta hai","oil mill",
 "food processing unit","cold drink ki factory","biscuit ki factory","atta ki dukan",
 "ration ki dukan","kirana store","general store","weekly market",
 "Mandi ka bhav","wholesale market","export import licence","customs duty",
 "shipping kaise hoti hai","cargo ship","container ship","port authority",
 "railway freight","goods train","truck ka transport","courier service",
 "DHL","Blue Dart","India Post","speed post","postal pin code",
 "bank chalu kaise kare","ATM se paisa nahi aa raha","debit card","credit card",
 "EMI card","loan closure","bank account se paise kaise nikale","ATM se paise kaise nikale",
 "UPI se paise bheje","UPI se paise mangaye","QR code scan","phone pe wallet",
 "digital wallet","Banking se fraud","fake call","OTP kya hai",
 "OTP kya bataye","SIM swap","cyber fraud se bache","password banaye",
 "two factor authentication","biometric login","face unlock","fingerprint login",
 "data privacy","personal data","Aadhaar se kya hota hai","Aadhaar leak",
 "Cyber security tips","phishing kya hai","malware kya hai","virus kaise aata hai",
 "phone hang ho gaya","phone slow ho gaya","storage full","RAM kya hai",
 "battery jaldi khatam","charger ka kaam nahi","screen crack","phone ki warranty",
 "laptop me problem","computer hang","printer kaam nahi kar raha","WiFi nahi chal raha",
 "router kya hai","modem kya hai","signal nahi aa raha","network problem",
 "tower kya hota hai","mobile tower radiation","signal strength","airplane mode",
 "sim card change","e-sim kya hai","5G kya hai","4G vs 5G",
 "jio","airtel","vi","bsnl","Airtel ka plan","recharge kaise kare",
 "bill payment","auto pay","dhan se recharge","free recharge offer",
 "membership kya hai","credit card se cash","EMI card kya hai","loan se ghar",
 "ghar ka EMI kitna","car ka EMI","bike ka EMI","vehicle finance",
 "car insurance","bike insurance","third party insurance","claim kaise kare",
 "accident hua toh kya kare","police complaint","FIR kahan likhe",
 "court case kaise kare","vakil ki fees","Nyaya kya hai","kanon kya hai",
 "Article 21","fundamental rights","constitutional rights","loktantra kya hai",
 "democracy kya hai","election commission","voting ki age","NOTA kya hai",
 "EVM kya hai","election fraud","voter list","EPIC card",
 "religion ka azaad","secularism","communal harmony","peaceful coexistence",
 "sahyadri ki tale","Himalaya ki choti","Rivers of India","Ganga ka uptoar",
 "Ganga pollution","Yamuna pollution","Narmada","Godavari","Krishna river",
 "Kaveri river","Tapti","Mahanadi","Son river","Ghaghara river",
 "Dam kya hota hai","Bhakra Nangal","Hirakud dam","Sardar Sarovar",
 "Tehri dam","bhedne ka kaaran","landslide","avalanche","earthquake kitne aate hain",
 "seismic zone","seismometer","Richter scale","magnitude kya hota hai",
 "bhukamp se bache","earthquake safety","fire safety","aag se bache",
 "dhamaka","blast kya hota hai","explosion kya hota hai","bomb blast se bache",
 "airport pe kya kare","security check kya hai","visa interview","travel document",
 "travel insurance","luggage me kya na rakhе","passport banwana",
 "pin code kya hota hai","village me kya kya hota hai","gaon me rozana ka kaam",
 "khet se kya aata hai","kisan ki majdoori","majdoor ki kasmaat","bijli se kheti",
 "diesel pump","kisan ka loan","kisan ko kya milta hai","subsidy kaise le",
 "pradhan mantri yojana","sarkar ki scheme","yojana ka labh","apply kaise kare",
 "online apply","form bharna","document lagana","signature kahan kare",
 "Aadhaar se verify","OTP verify","bank se naam badalna","name mismatch fix",
 "account freeze ho gaya","KYC kya hai","Aadhaar se bank me naam","PAN card",
 "PAN se bank me naam","EPFO","EPF kya hai","provident fund",
 "pension ke liye kya kare","PPF kya hai","NPS kya hai","NPS me kitna invest kare",
 "SIP kya hai","mutual fund kya hai","nivesh kya hai","share kya hota hai",
 "stock market crash","bull market","bear market","BSE NSE kya hai",
 "IPO kya hai","IPO me paisa lagana","demat account","trading kya hai",
 "sebi kya hai","RBI kya hai","bank ka nationalization","private bank",
 "public sector bank","co-operative bank","NBFC kya hai","microfinance",
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
