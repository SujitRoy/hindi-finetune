#!/usr/bin/env python3
"""Notebooks must declare the accelerator the user actually wants: GPU T4 x2.

Kaggle shows "The notebook you're importing contains new settings" when the
imported file's settings DIFFER from the kernel's. So the fix is not to strip
metadata.kaggle - stripping it makes the notebook imply None, and accepting the
dialog then silently strips the GPU. Declaring GPU T4 x2 makes the comparison a
no-op and is the only setting that is both safe and quiet.

This test previously asserted the opposite and failed the moment the reasoning
was corrected.
"""
import json, glob, sys
WANT = "GPU T4 x2"
bad = []
for p in sorted(glob.glob("notebooks/*.ipynb")) + ["hinglish-sft-lfm25-1.2b.ipynb"]:
    m = json.load(open(p)).get("metadata", {})
    k = m.get("kaggle", {})
    acc = k.get("accelerator")
    stray = [x for x in ("accelerator", "colab") if x in m]
    if acc != WANT or stray:
        bad.append(f"{p}: accelerator={acc!r} stray={stray}")
for b in bad:
    print(b)
print("=" * 66)
print(f"  FAIL - notebook would ask to change the accelerator" if bad
      else f"  PASS - every notebook declares {WANT}; imports will not change it")
sys.exit(1 if bad else 0)
