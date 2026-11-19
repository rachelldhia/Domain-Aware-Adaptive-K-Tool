"""Frozen text encoders. All outputs are L2-normalised, so dot product = cosine."""
from __future__ import annotations
import re, zlib
import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class SentenceTransformerEncoder:
    def __init__(self, model: str, revision: str | None = None):
        from sentence_transformers import SentenceTransformer
        rev = None if (revision or "").startswith("PIN_THIS") else revision
        self.name = f"{model}@{rev or 'UNPINNED'}"
        self.model = SentenceTransformer(model, revision=rev)

    def encode(self, texts: list[str]) -> np.ndarray:
        e = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return _norm(np.asarray(e, dtype=np.float64))


class HashingEncoder:
    """Dependency-free bag-of-words encoder. ONLY for --mock pipeline testing."""
    def __init__(self, dim: int = 1024):
        self.dim, self.name = dim, f"hashing-{dim}(MOCK)"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", t.lower())
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            for f in feats:
                out[i, zlib.crc32(f.encode()) % self.dim] += 1.0
        return _norm(np.log1p(out))


def make_encoder(cfg: dict, mock: bool = False):
    if mock or cfg.get("type") == "hashing":
        return HashingEncoder()
    return SentenceTransformerEncoder(cfg["model"], cfg.get("revision"))
