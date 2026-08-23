"""minilm.py: sentence embeddings with sentence-transformers/all-MiniLM-L6-v2.

Same pipeline as the model's modules.json: transformer, mean pooling over non-padding tokens, L2
normalisation, inputs truncated to 256 tokens. Uses transformers directly so it doesn't need the
sentence-transformers package.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


class MiniLM:
    def __init__(self, device: str | None = None, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(NAME)
        # fp32 on every device: the published corpora were embedded in fp32.
        self.model = AutoModel.from_pretrained(NAME).eval().to(self.device)
        self.max_length = max_length

    def encode(self, texts: List[str], batch: int = 256) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = self.tok(texts[i:i + batch], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**b).last_hidden_state.float()
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, DIM), np.float32)


def batched(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
