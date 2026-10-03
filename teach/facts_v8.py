#!/usr/bin/env python3
"""Fact seeds: question + a gold substring the answer MUST contain. Verification, not judgment.

Why not the teacher-as-judge. Calibrated on this box: known-evasive v7 rows scored 3-4 and
clean rows scored 3-5 (full overlap), and it called broken verb agreement
(करते हैं -> करते है, के लिए -> का लिए) indistinguishable from correct in 2 of 3 pairs. The
teacher is a private preview with no published baseline and it confabulates on its own - it
answered "भारत की राजधानी" with "मुख्य शहर बोलपुर". A judge that cannot separate good from bad
is a random filter that costs 6-8 hours to prove it is noise. The other configured providers
refuse script callers (HTTP 401 "unauthorized client detected"), so a stronger judge is not
reachable from here.

A substring check cannot be fooled. The row is kept only if the answer contains the fact, so
these rows teach something specific and checkable. ~100 seeds x both scripts = ~200 rows:
too few to make a 1.2B knowledgeable, enough to stop it inventing a capital.

  python3 teach/facts_v8.py                       # writes teach/prompts_facts_v8.jsonl

Each prompt carries "gold": [...] and reg="fact", which generate.py turns into a hard
containment check on the answer.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "prompts_facts_v8.jsonl")

# (Devanagari question, [acceptable gold substrings], romanized Hinglish question)
FACTS = [
 ("भारत की राजधानी कौन सी है?", ["नई दिल्ली"], "India ki rajdhani kaun si hai?"),
 ("भारत में कुल कितने राज्य हैं?", ["28"], "India mein kitne state hain?"),
 ("भारत में कितने केंद्र शासित प्रदेश हैं?", ["8"], "India mein kitne UT hain?"),
 ("सौर मंडल में कितने ग्रह हैं?", ["8"], "Solar system mein kitne planet hain?"),
 ("पानी का रासायनिक सूत्र क्या है?", ["H2O", "H₂O"], "Water ka chemical formula kya hai?"),
 ("लवण का रासायनिक नाम क्या है?", ["सोडियम क्लोराइड"], "Namak ka chemical naam kya hai?"),
 ("ताजमहल कहाँ स्थित है?", ["आगरा"], "Taj Mahal kahan hai?"),
 ("ताजमहल किसने बनवाया?", ["शाहजहाँ"], "Taj Mahal kisne banwaya?"),
 ("भारत के पहले प्रधानमंत्री कौन थे?", ["जवाहरलाल नेहरू"], "India ke pehle PM kaun the?"),
 ("भारत की पहली महिला प्रधानमंत्री कौन थीं?", ["इंदिरा गांधी"], "India ki pehli mahila PM kaun thi?"),
 ("भारत को आज़ादी कब मिली?", ["1947"], "India ko aazadi kab mili?"),
 ("महात्मा गांधी का पूरा नाम क्या था?", ["मोहनदास"], "Gandhi ji ka pura naam kya tha?"),
 ("महात्मा गांधी का जन्म कब हुआ?", ["1869"], "Gandhi ji ka janm kab hua?"),
 ("सूर्य के सबसे नज़दीक कौन सा ग्रह है?", ["बुध"], "Sun ke sabse paas kaunsa planet hai?"),
 ("सबसे बड़ा ग्रह कौन सा है?", ["बृहस्पति", "वृहस्पति", "Jupiter"], "Sabse bada planet kaunsa hai?"),
 ("मानव शरीर में कितनी हड्डियाँ होती हैं?", ["206"], "Human body mein kitni haddiyan hoti hain?"),
 ("मानव शरीर का सबसे बड़ा अंग कौन सा है?", ["त्वचा"], "Body ka sabse bada organ kaunsa hai?"),
 ("खून का रंग लाल क्यों होता है?", ["हीमोग्लोबिन"], "Khoon laal kyun hota hai?"),
 ("दिल में कितने कक्ष होते हैं?", ["4"], "Heart mein kitne chambers hote hain?"),
 ("प्रकाश की गति लगभग कितनी होती है?", ["3 लाख", "3,00,000", "300000"], "Light ki speed kitni hoti hai?"),
 ("भारत की मुद्रा क्या है?", ["रुपया", "रुपये"], "India ki currency kya hai?"),
 ("भारतीय रिज़र्व बैंक की स्थापना कब हुई?", ["1935"], "RBI ki sthapna kab hui?"),
 ("गणतंत्र दिवस कब मनाया जाता है?", ["26 जनवरी"], "Republic Day kab manate hain?"),
 ("स्वतंत्रता दिवस कब मनाया जाता है?", ["15 अगस्त"], "Independence Day kab manate hain?"),
 ("भारत का राष्ट्रीय पशु कौन सा है?", ["बाघ", "व्याघ्र"], "India ka national animal kaunsa hai?"),
 ("भारत का राष्ट्रीय पक्षी कौन सा है?", ["मोर"], "India ka national bird kaunsa hai?"),
 ("भारत का राष्ट्रीय फूल कौन सा है?", ["कमल"], "India ka national flower kaunsa hai?"),
 ("भारत का राष्ट्रीय गान कौन सा है?", ["जन गण मन"], "India ka national anthem kaunsa hai?"),
 ("भारत की राजभाषा क्या है?", ["हिन्दी", "हिंदी"], "India ki rajbhasha kya hai?"),
 ("क्रिकेट विश्व कप 2023 किसने जीता?", ["भारत"], "Cricket World Cup 2023 kisne jeeta?"),
 ("भारत की सबसे लंबी नदी कौन सी है?", ["गंगा"], "India ki sabse lambi river kaunsi hai?"),
 ("सबसे ऊँचा पर्वत शिखर कौन सा है?", ["एवरेस्ट", "सगरमाथा"], "Sabse uncha pahad kaunsa hai?"),
 ("थर्मोमीटर किसे मापने के लिए उपयोग होता है?", ["तापमान", "ताप"], "Thermometer kise mapne ke liye use hota hai?"),
 ("टेलीफोन का आविष्कार किसने किया?", ["ग्राहम बेल"], "Telephone kisne invent kiya?"),
 ("बल्ब का आविष्कार किसने किया?", ["एडिसन"], "Bulb kisne invent kiya?"),
 ("कंप्यूटर के जनक कौन कहे जाते हैं?", ["चार्ल्स बैबेज", "बैबेज"], "Computer ka father kise kehte hain?"),
 ("भारतीय संविधान कब लागू हुआ?", ["1950"], "Indian Constitution kab lagu hua?"),
 ("संसद के दोनों सदन क्या हैं?", ["लोक सभा", "राज्य सभा"], "Parliament ke dono house kya hain?"),
 ("सूर्य ग्रहण किस पक्ष में लगता है?", ["अमावस्या"], "Solar eclipse kab lagta hai?"),
 ("चंद्र ग्रहण किस पक्ष में लगता है?", ["पूर्णिमा"], "Lunar eclipse kab lagta hai?"),
 ("पृथ्वी सूर्य का एक चक्कर कितने दिन में लगाती है?", ["365"], "Earth ka ek chakkar kitne din mein lagta hai?"),
 ("विश्व का सबसे बड़ा महासागर कौन सा है?", ["प्रशान्त", "पैसिफिक"], "Sabse bada ocean kaunsa hai?"),
 ("क्षेत्रफल में भारत का सबसे बड़ा राज्य कौन सा है?", ["राजस्थान"], "Area mein India ka sabse bada state kaunsa hai?"),
 ("भारत का सबसे छोटा राज्य कौन सा है?", ["गोवा"], "India ka sabse chhota state kaunsa hai?"),
 ("मध्य प्रदेश की राजधानी क्या है?", ["भोपाल"], "MP ki rajdhani kya hai?"),
 ("महाराष्ट्र की राजधानी क्या है?", ["मुंबई"], "Maharashtra ki rajdhani kya hai?"),
 ("तमिलनाडु की राजधानी क्या है?", ["चेन्नई"], "Tamil Nadu ki rajdhani kya hai?"),
 ("पंजाब की राजधानी क्या है?", ["चंडीगढ़"], "Punjab ki rajdhani kya hai?"),
 ("कर्नाटक की राजधानी क्या है?", ["बेंगलुरु", "बेंगलोर"], "Karnataka ki rajdhani kya hai?"),
 ("गुजरात की राजधानी क्या है?", ["गांधीनगर"], "Gujarat ki rajdhani kya hai?"),
 ("उत्तर प्रदेश की राजधानी क्या है?", ["लखनऊ"], "UP ki rajdhani kya hai?"),
 ("बिहार की राजधानी क्या है?", ["पटना"], "Bihar ki rajdhani kya hai?"),
 ("पश्चिम बंगाल की राजधानी क्या है?", ["कोलकाता"], "West Bengal ki rajdhani kya hai?"),
 ("जल का हिमांक कितना होता है?", ["0"], "Paani ka freezing point kitna hota hai?"),
 ("जल का क्वथनांक कितना होता है?", ["100"], "Paani ka boiling point kitna hota hai?"),
 ("वायु में सबसे अधिक पाई जाने वाली गैस कौन सी है?", ["नाइट्रोजन"], "Hawa mein sabse jyada kaunsi gas hoti hai?"),
 ("सबसे हल्की गैस कौन सी है?", ["हाइड्रोजन"], "Sabse halki gas kaunsi hai?"),
 ("पौधे भोजन बनाने की किस प्रक्रिया का उपयोग करते हैं?", ["प्रकाश संश्लेषण", "photosynthesis"], "Plants food banane ke liye kaunsi process use karte hain?"),
 ("विटामिन डी हमें मुख्यतः किससे मिलता है?", ["सूर्य"], "Vitamin D kahan se milta hai?"),
 ("रक्तहीनता किस तत्व की कमी से होती है?", ["लोहा", "आयरन"], "Anemia kis kami se hota hai?"),
 ("क्षय रोग किस अंग पर आक्रमण करता है?", ["फेफड़", "फेफड़ा"], "TB kis organ par attack karta hai?"),
 ("डायबिटीज़ किस अंग से संबंधित रोग है?", ["अग्न्याशय", "पैन्क्रियाज"], "Diabetes kis organ se juda hai?"),
 ("HIV किस रोग का कारण बनता है?", ["एड्स"], "HIV kis rog ka karan banta hai?"),
 ("श्वसन के लिए मनुष्य किस अंग का उपयोग करता है?", ["फेफड़", "फेफड़ा"], "Saans lene ke liye kaunsa organ use hota hai?"),
 ("आमाशय में कौन सा अम्ल स्रावित होता है?", ["हाइड्रोक्लोरिक", "HCl"], "Stomach mein kaunsa acid hota hai?"),
 ("भारत का पहला उपग्रह कौन सा था?", ["आर्यभट्ट"], "India ka pehla satellite kaunsa tha?"),
 ("चंद्रमा पर उतरने वाले पहले व्यक्ति कौन थे?", ["नील आर्मस्ट्रांग"], "Chand par pehle utarne wale kaun the?"),
 ("इसरो का मुख्यालय कहाँ है?", ["बेंगलुरु", "बेंगलोर"], "ISRO ka headquarters kahan hai?"),
 ("भारत का पहला प्रक्षेपण केंद्र कहाँ है?", ["श्रीहरिकोटा"], "India ka pehla launch station kahan hai?"),
 ("GST भारत में कब लागू हुआ?", ["2017"], "GST India mein kab lagu hua?"),
 ("आधार कार्ड किस संस्था द्वारा जारी किया जाता है?", ["यूआईडीएआई", "UIDAI"], "Aadhaar card kaunsi agency jari karti hai?"),
 ("अंतर्राष्ट्रीय योग दिवस कब मनाया जाता है?", ["21 जून"], "Yoga Day kab manate hain?"),
 ("विश्व पर्यावरण दिवस कब मनाया जाता है?", ["5 जून"], "Environment Day kab manate hain?"),
 ("सबसे कठोर प्राकृतिक पदार्थ कौन सा है?", ["हीरा"], "Sabse sakht natural cheez kaunsi hai?"),
 ("कमरे के ताप पर द्रव अवस्था में रहने वाला धातु कौन सा है?", ["पारा", "मरकरी"], "Room temperature mein liquid metal kaunsa hai?"),
 ("न्यूटन का गति का तीसरा नियम क्या कहता है?", ["क्रिया", "प्रतिक्रिया"], "Newton ka third law kya kehta hai?"),
 ("गुरुत्वाकर्षण का सार्वत्रिक नियम किसने दिया?", ["न्यूटन"], "Gravity ka niyam kisne diya?"),
 ("बंगलौर का नया नाम क्या है?", ["बेंगलुरु"], "Bangalore ka naya naam kya hai?"),
 ("स्वर्ण मंदिर कहाँ स्थित है?", ["अमृतसर"], "Golden temple kahan hai?"),
 ("लाल किला कहाँ स्थित है?", ["दिल्ली"], "Red fort kahan hai?"),
 ("हवामहल कहाँ स्थित है?", ["जयपुर"], "Hawa mahal kahan hai?"),
 ("काशी विश्वनाथ मंदिर कहाँ है?", ["वाराणसी"], "Kashi vishwanath temple kahan hai?"),
 ("भारत का सबसे व्यस्त रेलवे स्टेशन कौन सा है?", ["छत्रपति शिवाजी", "सीएसटी", "CST"], "India ka busiest railway station kaunsa hai?"),
 ("भारत में पहली मेट्रो कहाँ चली?", ["दिल्ली"], "Metro India mein pehli baar kahan chala?"),
 ("भारत का पहला IIT कहाँ स्थापित हुआ?", ["खड़गपुर"], "Pehla IIT kahan bana?"),
 ("नोबेल पुरस्कार पाने वाले पहले भारतीय कौन हैं?", ["रवींद्रनाथ टागोर", "टागोर"], "Pehle Indian Nobel winner kaun the?"),
 ("भारत से नोबेल पाने वाली पहली महिला कौन थीं?", ["मदर टेरेसा"], "Indian woman Nobel prize pehli baar kisne jeeta?"),
 ("व्हाट्सएप किस कंपनी का स्वामित्व है?", ["मेटा", "फेसबुक"], "WhatsApp kis company ka hai?"),
 ("गूगल के सह-संस्थापक कौन हैं?", ["पेज", "ब्रिन"], "Google ke co-founder kaun hain?"),
 ("मानव में सामान्य रक्तचाप लगभग कितना होता है?", ["120", "12"], "Normal blood pressure kitna hota hai?"),
]

def key(prefix, text):
    return hashlib.sha1((prefix + text).encode("utf-8")).hexdigest()[:16]

if __name__ == "__main__":
    seen, out = set(), []
    for hi_q, golds, hg_q in FACTS:
        for plang, q in (("hi", hi_q.strip()), ("hinglish", hg_q.strip())):
            k = key(plang, q)
            if k in seen or not golds: continue
            seen.add(k)
            out.append({"key": k, "instruction": q, "plang": plang, "src": "facts_v8",
                        "reg": "fact", "topic": "fact", "gold": golds,
                        "budget_force": "short"})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(out):,} verifiable fact prompts "
          f"({sum(1 for r in out if r['plang']=='hi'):,} hi, "
          f"{sum(1 for r in out if r['plang']=='hinglish'):,} hinglish)")
