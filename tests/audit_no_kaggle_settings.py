#!/usr/bin/env python3
"""The notebooks must NOT carry a Kaggle settings block.

metadata.kaggle.accelerator makes every import pop Kaggle's
"The notebook you're importing contains new settings" dialog and overwrite
whatever the user had selected. That happened on every single run. The
accelerator is set once in the UI and is not the notebook's business.
"""
import json, glob, sys
bad = []
for p in sorted(glob.glob("notebooks/*.ipynb")) + ["hinglish-sft-lfm25-1.2b.ipynb"]:
    m = json.load(open(p)).get("metadata", {})
    if "kaggle" in m:
        bad.append(f"{p}: {m['kaggle']}")
print("\n".join(bad) if bad else "  no notebook carries a kaggle settings block")
print("=" * 62)
print("  FAIL - Kaggle will ask to overwrite settings on import" if bad
      else "  PASS - imports will not touch Accelerator")
sys.exit(1 if bad else 0)
