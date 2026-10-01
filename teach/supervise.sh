#!/bin/bash
# Keeps the corpus growing without supervision. Generation and prompt expansion run
# as separate jobs; this watches both, and when either finishes it starts the next
# stage. Prompt rounds hold their output in memory until they exit, so a round that
# dies loses everything it collected - this restarts rather than leaving a hole.
cd /home/ubuntu/hindi-finetune/teach
LOG=/tmp/supervise.log
say() { echo "[$(date '+%H:%M:%S')] $*" >> $LOG; }

rebuild_todo () {
  /usr/bin/python3.13 - << 'PY' >> $LOG 2>&1
import json, glob, os
seen=set(); out=[]
for f in ("prompts.jsonl","extra_prompts2.jsonl","extra_prompts3.jsonl","extra_prompts4.jsonl",
          "extra_prompts5.jsonl","extra_prompts6.jsonl","extra_prompts7.jsonl","pool_new.jsonl"):
    if not os.path.exists(f): continue
    for l in open(f,encoding="utf-8"):
        r=json.loads(l); k=r["instruction"].strip().lower()
        if k in seen or not r.get("plang"): continue
        seen.add(k); r["key"]=f"m:{len(seen)}"; out.append(r)
done=set()
for f in glob.glob("teacher_gen*.jsonl"):
    try:
        for l in open(f,encoding="utf-8"): done.add(json.loads(l)["instruction"].strip().lower())
    except Exception: pass
todo=[r for r in out if r["instruction"].strip().lower() not in done]
with open("pool_todo.jsonl","w",encoding="utf-8") as f:
    for r in todo: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"  pool {len(out)}  answered {len(done)}  todo {len(todo)}")
PY
}

rebuild_todo
say "supervisor start"

while true; do
  # answer generation: restart whenever the pool grows
  if ! pgrep -f "genmain.py" > /dev/null; then
    n=$(wc -l < pool_todo.jsonl 2>/dev/null || echo 0)
    if [ "$n" -gt 50 ]; then
      say "generation not running, $n todo - starting"
      rebuild_todo
      n=$(wc -l < pool_todo.jsonl)
      setsid nohup /usr/bin/python3.13 /tmp/genmain.py "$n" 12 >> /tmp/genmain.log 2>&1 < /dev/null &
    else
      say "nothing to generate ($n todo)"
    fi
  fi
  # prompt expansion rounds
  for r in 4 5; do
    if ! pgrep -f "expand_pool$r.py" > /dev/null && [ ! -f extra_prompts${r}p ].jsonl ]; then :; fi
  done
  sleep 120
done
