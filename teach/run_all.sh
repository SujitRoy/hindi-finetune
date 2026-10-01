#!/bin/bash
# Detached end-to-end: expand the prompt pool, merge it, generate both languages,
# then audit the result. Safe to re-run; generation resumes by prompt hash.
set -u
cd /home/ubuntu/hindi-finetune/teach
PY=/usr/bin/python3.13
TARGET="${1:-15000}"
CONC="${2:-16}"

echo "=== $(date +%H:%M:%S) step 1: expand the prompt pool ==="
$PY expand_pool.py 2>&1 

echo "=== $(date +%H:%M:%S) step 2: merge into prompts.jsonl ==="
$PY merge_pool.py 2>&1 

echo "=== $(date +%H:%M:%S) step 3: generate $TARGET rows at concurrency $CONC ==="
$PY generate.py "$TARGET" "$CONC" 2>&1 

echo "=== $(date +%H:%M:%S) step 4: audit ==="
$PY audit_final.py 2>&1 
echo "=== $(date +%H:%M:%S) done ==="
