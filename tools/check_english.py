#!/usr/bin/env python3
"""Fails if a tracked file, or a file name, contains CJK text.

The repository is kept in English only (see CONTRIBUTING.md). Run from the
repository root; CI runs it on every push and pull request.
"""

import re
import subprocess
import sys

# CJK punctuation and kana, ideographs, Hangul, full-width forms. Written as code points so that this
# file passes its own check.
RANGES = [(0x3000, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xFF00, 0xFFEF)]
CJK = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in RANGES) + "]")


def main():
    files = [f for f in subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n") if f]
    bad = []
    for name in files:
        if CJK.search(name):
            bad.append(f"{name}: file name")
        try:
            with open(name, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    if CJK.search(line):
                        bad.append(f"{name}:{n}")
                        break
        except (UnicodeDecodeError, IsADirectoryError):
            continue  # binary
    if bad:
        print("non-English text found:\n  " + "\n  ".join(bad))
        return 1
    print(f"{len(files)} files checked: English only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
