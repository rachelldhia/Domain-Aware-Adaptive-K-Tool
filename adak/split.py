"""Template-level validation/test split (stratified by domain group)."""
from __future__ import annotations
import numpy as np
from collections import defaultdict


def template_split(tasks, n_val: int, seed: int):
    groups = defaultdict(set)
    for t in tasks:
        groups[t.domains[0] if t.n_domains == 1 else "multi"].add(t.base_template)
    n_total = sum(len(v) for v in groups.values())
    raw = {g: n_val * len(v) / n_total for g, v in groups.items()}
    alloc = {g: int(np.floor(x)) for g, x in raw.items()}
    for g in sorted(raw, key=lambda g: raw[g] - alloc[g], reverse=True)[: n_val - sum(alloc.values())]:
        alloc[g] += 1
    rng, val = np.random.default_rng(seed), set()
    for g in sorted(groups):
        tpls = sorted(groups[g]); rng.shuffle(tpls)
        val.update(tpls[: alloc[g]])
    test = {t.base_template for t in tasks} - val
    return {"validation_templates": sorted(val), "test_templates": sorted(test), "seed": seed, "alloc": alloc}


def apply_split(tasks, split):
    v, t = set(split["validation_templates"]), set(split["test_templates"])
    assert not (v & t)
    return [x for x in tasks if x.base_template in v], [x for x in tasks if x.base_template in t]
