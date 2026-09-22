#!/usr/bin/env python3
"""Where each headline number is claimed, and in what words.

check_manuscript.py already asks whether a number matches the data. This asks
a different question: whether the same number is described the same way
everywhere it appears. A paper can be arithmetically right and still say two
different things about the same result in the abstract and the conclusion,
which is the kind of thing a reviewer reads as carelessness.

For each number the paper leans on, this prints every sentence that carries it
along with the section it sits in, so the wordings can be read side by side.
Nothing is checked automatically; the point is to put them next to each other.
"""
from __future__ import annotations

import pathlib
import re
import sys

TEX = pathlib.Path(__file__).resolve().parents[1] / "manuscript" / "main.tex"

# The numbers the argument rests on, with the shape they appear in.
CLAIMS = {
    "13.7": r"13\.7",
    "1.1 % (short-term rating)": r"\{1\.1\}\{\\percent\}",
    "2.6 % (reactive)": r"\{2\.6\}\{\\percent\}",
    "-2.7 to +0.9 (voltage)": r"-2\.7|\+0\.9",
    "56.5-110.4 MW reserve": r"56\.5|110\.4",
    "7243 MWh curtailed": r"7243",
    "8.5 % (RoCoF)": r"\{8\.5\}\{\\percent\}",
    "4.6 % (sync inertia)": r"\{4\.6\}\{\\percent\}",
    "ac violation": r"diagnostic|violation in every",
    "degenerate / subgradient": r"degenerate|subgradient",
}

SECTION = re.compile(r"\\(sub)*section\{([^}]*)\}")


def sections(text: str):
    """(start offset, label) for every sectioning command, in order."""
    found = [(m.start(), m.group(2)) for m in SECTION.finditer(text)]
    return found


def label_at(marks, offset: int) -> str:
    name = "front matter"
    for start, title in marks:
        if start <= offset:
            name = title
        else:
            break
    return name


def sentence_at(text: str, offset: int) -> str:
    start = text.rfind(". ", max(0, offset - 700), offset)
    start = start + 2 if start > 0 else max(0, offset - 300)
    end = text.find(". ", offset)
    end = end + 1 if end > 0 else min(len(text), offset + 300)
    return " ".join(text[start:end].split())


def main() -> None:
    text = TEX.read_text(encoding="utf-8")
    marks = sections(text)
    for name, pattern in CLAIMS.items():
        hits = list(re.finditer(pattern, text))
        print(f"\n{'=' * 74}\n{name}  —  {len(hits)}곳\n{'=' * 74}")
        seen = set()
        for hit in hits:
            where = label_at(marks, hit.start())
            said = sentence_at(text, hit.start())
            key = (where, said[:80])
            if key in seen:
                continue
            seen.add(key)
            print(f"[{where}]")
            print(f"   {said[:430]}")


if __name__ == "__main__":
    sys.exit(main())
