#!/bin/bash
# Progress check. Safe to run any time; read-only.
cd /home/ubuntu/hindi-finetune/teach
echo "processes:"
ps -eo pid,etime,pcpu,cmd | grep -E "run_all|expand_pool|generate" | grep -v grep | sed 's/^/  /'
echo
echo "log tail:"
tail -6 /tmp/runall.log | sed 's/^/  /'
echo
for f in prompts.jsonl extra_prompts.jsonl teacher_gen.jsonl; do
  [ -f "$f" ] && printf "  %-22s %6s rows\n" "$f" "$(wc -l < $f)"
done
echo
if [ -f teacher_gen.jsonl ]; then
  /usr/bin/python3.13 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open("teacher_gen.jsonl",encoding="utf-8")]
c=collections.Counter(r["lang"] for r in rows)
w=sorted(len(r["response"].split()) for r in rows); n=len(w)
print(f"  generated: {n:,}  hindi {c['hindi']:,}  hinglish {c['hinglish']:,}")
print(f"  words median {w[n//2]}  p10 {w[n//10]}")
PY
fi
