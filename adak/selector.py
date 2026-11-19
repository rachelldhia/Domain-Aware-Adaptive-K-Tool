"""Domain-Aware Adaptive-k selector (Section 3.2.5 of the proposal)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .data import Tool


@dataclass(frozen=True)
class Params:
    epsilon: float
    tau: float
    min_k: int
    max_k: int
    def asdict(self): return asdict(self)


@dataclass
class Selection:
    domains: list[str]           # D*(q)
    tool_ids: list[str]          # S(q) in catalogue order (incl. U)
    pool_size: int               # |P(q)|
    flags: list[str]             # e.g. 'pool_smaller_than_min_k'


class DomainAwareAdaptiveK:
    def __init__(self, tools: list[Tool], encoder):
        self.tools = tools                                    # catalogue order = fixed identifier order
        self.encoder = encoder
        self.shared = [t for t in tools if t.domain is None]
        self.cand = [t for t in tools if t.domain is not None]   # tools that get scored
        self.domains = sorted({t.domain for t in self.cand})
        self.E = encoder.encode([t.text for t in self.cand])     # (|cand|, d), normalised
        d_idx = {d: i for i, d in enumerate(self.domains)}
        self.tool_dom = np.array([d_idx[t.domain] for t in self.cand])
        C = np.stack([self._unit(self.E[self.tool_dom == i].mean(0)) for i in range(len(self.domains))])
        self.C = C                                              # domain centroids (shared excluded)
        self.cat_order = {t.id: i for i, t in enumerate(tools)}
        self.def_tokens = {t.id: t.def_tokens for t in tools}

    @staticmethod
    def _unit(v): return v / max(np.linalg.norm(v), 1e-12)

    # ---- scoring (done once per query, reused over the whole parameter grid)
    def score(self, queries: list[str]):
        Q = self.encoder.encode(queries)
        return Q @ self.C.T, Q @ self.E.T        # A: (n, |D|)   S: (n, |cand|)

    def check(self, p: Params):
        if not (len(self.shared) <= p.min_k <= p.max_k <= len(self.tools)):
            raise ValueError(f"need |U|<=min_k<=max_k<=|T|, got {p} with |U|={len(self.shared)}, |T|={len(self.tools)}")

    # ---- selection
    def select(self, a: np.ndarray, s: np.ndarray, p: Params) -> Selection:
        self.check(p)
        flags = []
        sel_dom = np.where(a >= a.max() - p.epsilon - 1e-12)[0]          # Stage I
        pool = np.where(np.isin(self.tool_dom, sel_dom))[0]
        ids = [self.cand[i].id for i in pool]
        order = sorted(range(len(pool)), key=lambda j: (-round(float(s[pool[j]]), 12), ids[j]))
        ranked = [pool[j] for j in order]
        n_u = len(self.shared)
        chosen = [i for i in ranked if s[i] >= p.tau]                     # Stage II
        if n_u + len(chosen) < p.min_k:
            need = p.min_k - n_u
            if len(ranked) < need: flags.append("pool_smaller_than_min_k")
            chosen = ranked[:need]
        if n_u + len(chosen) > p.max_k:
            chosen = chosen[: p.max_k - n_u]
            flags.append("truncated_to_max_k")
        tool_ids = [t.id for t in self.shared] + [self.cand[i].id for i in chosen]
        tool_ids.sort(key=self.cat_order.get)
        return Selection([self.domains[i] for i in sel_dom], tool_ids, len(pool) + n_u, flags)

    def definition_tokens(self, tool_ids): return sum(self.def_tokens[i] for i in tool_ids)
    @property
    def full_catalogue_tokens(self): return sum(self.def_tokens.values())
