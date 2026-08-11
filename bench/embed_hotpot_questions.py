"""embed_hotpot_questions.py: real query vectors.

Until now every benchmark used held-out corpus vectors as queries, but real queries are questions.
This embeds the 1,000 HotpotQA questions exported by the old harness with the same model as the
corpus (all-MiniLM-L6-v2: mean pooling, then L2 normalisation, as in its 1_Pooling config).

Uses transformers directly because `import sentence_transformers` segfaults on this machine (exit
139). The pooling is the same.

    HF_HUB_OFFLINE=1 python bench/embed_hotpot_questions.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "_archive", "legacy_python", "hotpot_code", "benchmark_queries_export.json")
OUT = os.path.join(ROOT, "bench", "_lab_cache", "hotpot_questions_minilm.npy")
NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    qs = [q["question"] for q in json.load(open(SRC))]
    tok = AutoTokenizer.from_pretrained(NAME)
    mdl = AutoModel.from_pretrained(NAME).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(qs), 128):
            b = tok(qs[i:i + 128], padding=True, truncation=True, max_length=256,
                    return_tensors="pt")
            h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(e, dim=1).numpy())
    E = np.concatenate(out).astype(np.float32)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, E)
    print(f"{E.shape} -> {OUT}")


if __name__ == "__main__":
    main()
