"""Test accept() alone against the exact failure modes from the v1 run."""
import re, sys
import textwrap
src = open("gencell.py").read()
# exec ONLY the lines from the BAD constant through the end of accept()
block = src[src.index("    BAD   = re.compile"):src.index("    CACHED =")]
ns = {"re": __import__("re")}
exec(textwrap.dedent(block), ns)
accept = ns["accept"]
import re as _re
DEV=lambda s: sum("ऀ"<=c<="ॿ" for c in s)/max(1,len(s))
cases = [
 ("good deva",  "भारत की राजधानी नई दिल्ली है और यह उत्तर भारत के मैदानी क्षेत्र में स्थित है। यह शहर देश की राजनीतिक और आर्थिक गतिविधियों का केंद्र है और यहाँ बहुत से लोग रहते हैं। यह शहर उत्तर भारत के मैदानी भाग में स्थित है।", True),
 ("TRUNCATED", "भारत की राजधानी नई दिल्ली है और यह उत्तर भारत के मैदानी क्षेत्र में स्थित है तथा यह शहर देश की राजनीतिक गतिविधियों का प्रमुख केंद्र माना जाता है जहाँ संसद भवन स्थित", True),
 ("TRUNC hing","Neeche chaaro rakshak hain jo aapki madad karte hain aur ye sab aapko bahut fayda dete hain. Pehla apna swar bharta hai aur doosra apne parivaar ko sambhalta hai. Teesra bhookha aurtees ko deta hai aur chautha gusse", False),
 ("good hing", "Neeche chaaro rakshak hain jo aapki madad karte hain. Pehla apna swar bharta hai. Doosra apne parivaar ko sambhalta hai. Teesra bhookha aurtees ko deta hai. Chautha gusse mein sabko hata deta hai.", False),
 ("short",     "नई दिल्ली।", True),
 ("repetitive","लंबा जवाब " * 8 + "लंबा जवाब " * 8 + "लंबा जवाब।", True),
 ("english",   "The capital of India is New Delhi which is located in the northern part of the country and it is also the political and economic centre of everything that happens here today.", False),
]
print(f"{'case':12s} {'want_dev':>9s}  result")
print("-"*46)
fails=0
for name, txt, wd in cases:
    ok, why = accept(txt, wd)
    print(f"{name:12s} {str(wd):>9s}  {'ACCEPT' if ok else 'reject:'+why}")
    if name=="TRUNCATED" and ok: fails+=1; print("   !! truncated text was accepted")
    if name=="good deva" and not ok: fails+=1; print("   !! good text was rejected")
    if name=="good hing" and not ok: fails+=1; print("   !! good hinglish rejected")
    if name=="short" and ok: fails+=1; print("   !! short text accepted")
    if name=="TRUNCATED" and (ok or why!="truncated"): fails+=1; print(f"   !! long truncated: accepted={ok} why={why} (want reason 'truncated')")
    if name=="TRUNC hing" and (ok or why!="truncated"): fails+=1; print(f"   !! trunc hinglish: accepted={ok} why={why} (want reason 'truncated')")
print(f"\n{'ALL PASS' if fails==0 else str(fails)+' FAILURES'}")
