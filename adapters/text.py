from .base import BaseAdapter
from ..core import OperatorFamily
import numpy as np


class TextAdapter(BaseAdapter):
    """
    Text → co-occurrence operator → SFT.

    Operator: normalised word co-occurrence Laplacian.
    Parameters: per-word importance weights.
    Basis: diagonal perturbation per word.

    Example
    -------
    >>> doc = TextAdapter("hello world hello")
    >>> W = doc.kernel
    >>> lam = doc.predict(association_changes)
    """

    def __init__(self, texts: str | list[str],
                 max_words: int = 500, window: int = 5):
        if isinstance(texts, str):
            texts = [texts]
        self.texts = texts
        self.max_words = max_words
        self.window = window
        self._build()

    def _build(self):
        from collections import Counter
        all_tokens = []
        for text in self.texts:
            all_tokens.extend(text.lower().split())
        counter = Counter(all_tokens)
        vocab = [w for w, _ in counter.most_common(self.max_words)]
        n = len(vocab)
        self.vocab = vocab
        self.n_words = n

        word_to_idx = {w: i for i, w in enumerate(vocab)}

        cooc = np.zeros((n, n))
        for text in self.texts:
            tokens = [word_to_idx.get(w, -1) for w in text.lower().split()]
            tokens = [t for t in tokens if t >= 0]
            for i, ti in enumerate(tokens):
                for j in range(max(0, i - self.window),
                               min(len(tokens), i + self.window + 1)):
                    if i != j:
                        cooc[ti, tokens[j]] += 1

        self.cooc = cooc
        deg = np.sum(cooc, axis=1) + 1e-10
        A0 = np.diag(1.0 / np.sqrt(deg)) @ cooc @ np.diag(1.0 / np.sqrt(deg))
        A0 = (A0 + A0.T) / 2

        pairs = []
        for u in range(n):
            for v in range(u, n):
                if cooc[u, v] > 0:
                    pairs.append((u, v))
        max_pairs = min(len(pairs), 200)
        pairs = sorted(pairs, key=lambda uv: -cooc[uv[0], uv[1]])[:max_pairs]

        M = 1 + len(pairs)
        basis_arr = np.zeros((M, n, n))
        basis_arr[0] = np.eye(n)
        for k, (u, v) in enumerate(pairs):
            basis_arr[k + 1, u, v] = 1.0
            basis_arr[k + 1, v, u] = 1.0

        self._family = OperatorFamily(A0, list(basis_arr))
