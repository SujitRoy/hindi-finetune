#!/usr/bin/env python3
"""Teacher API config. Nothing sensitive is in this file, and nothing else reads secrets.

The baseUrl, apiKey and model id live in teach/teacher.local.json (gitignored, mode 600,
created once by hand). This repo is public and the teacher is a private preview model on
someone else's plan, so the id and the endpoint must never appear in a tracked file, a
commit message, a log line or a dataset card. Override the path with TEACHER_CONFIG.

  from teacher_cfg import TEACHER
  TEACHER["baseUrl"], TEACHER["apiKey"], TEACHER["model"]
"""
import json, os

_PATH = os.environ.get("TEACHER_CONFIG",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "teacher.local.json"))

def load(path=None):
    p = path or _PATH
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(
            f"teacher config not found at {p}\n"
            '  create it:  {"baseUrl": "...", "apiKey": "...", "model": "..."}\n'
            f"  then:       chmod 600 {p}")
    for k in ("baseUrl", "apiKey", "model"):
        if not d.get(k):
            raise SystemExit(f"teacher config {p} is missing '{k}'")
    return d

TEACHER = load()

def redact(text):
    """Strip the key and the model id from anything destined for a log or a commit."""
    if not isinstance(text, str):
        text = str(text)
    for secret in (TEACHER["apiKey"], TEACHER["model"], TEACHER["baseUrl"]):
        if secret:
            text = text.replace(secret, "<teacher>")
    return text

if __name__ == "__main__":
    # prints presence and lengths only, never the values
    print("baseUrl len", len(TEACHER["baseUrl"]),
          "| apiKey len", len(TEACHER["apiKey"]),
          "| model len", len(TEACHER["model"]),
          "| config", _PATH)
