"""embed_hotpot_questions.py: real query vectors.

Held-out passages are the usual ANN-benchmark queries, but a retrieval system serves questions.
These 1,000 HotpotQA questions (hotpot_questions.json; HotpotQA is CC BY-SA 4.0) are embedded with
the same model as the corpus so benchmarks can report both query sets.

    python bench/data/embed_hotpot_questions.py
        -> bench/data/hotpot_questions_minilm.npy
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from minilm import MiniLM  # noqa: E402

SRC = os.path.join(HERE, "hotpot_questions.json")
OUT = os.path.join(HERE, "hotpot_questions_minilm.npy")


def main():
    qs = [q["question"] for q in json.load(open(SRC, encoding="utf-8"))]
    E = MiniLM().encode(qs, batch=128)
    np.save(OUT, E)
    print(f"{E.shape} -> {OUT}")


if __name__ == "__main__":
    main()
