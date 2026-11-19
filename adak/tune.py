"""Validation-only parameter search (Section 3.2.6). Never touches test tasks."""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from .selector import Params


def task_row(sel_obj, task, selection):
    S = set(selection.tool_ids)
    dtrue = set(task.domains)
    wr = set(task.write_tools)
    return {
        "n_tools": len(S),
        "def_tokens": sel_obj.definition_tokens(selection.tool_ids),
        "domain_recall": len(dtrue & set(selection.domains)) / len(dtrue),
        "complete_domain": float(dtrue <= set(selection.domains)),
        "write_cov": (len(wr & S) / len(wr)) if wr else np.nan,
        "complete_write": (float(wr <= S)) if wr else np.nan,
    }


def offline_metrics(sel_obj, tasks, A, S, p: Params) -> dict:
    rows = [task_row(sel_obj, t, sel_obj.select(A[i], S[i], p)) for i, t in enumerate(tasks)]
    df = pd.DataFrame(rows)
    out = df.mean(numeric_only=True).to_dict()
    out["token_ratio"] = out["def_tokens"] / sel_obj.full_catalogue_tokens
    return out


def build_grid(sel_obj, A, S, cfg: dict) -> list[Params]:
    """eps from the observed gap distribution, tau from the query-tool score distribution."""
    gaps = (A.max(1, keepdims=True) - A).ravel()
    gaps = gaps[gaps > 0]
    eps = sorted({0.0, *np.quantile(gaps, cfg["epsilon_quantiles"]).round(4).tolist()})
    taus = sorted(set(np.quantile(S.ravel(), cfg["tau_quantiles"]).round(4).tolist()))
    grid = []
    for e, t, lo, hi in itertools.product(eps, taus, cfg["min_k"], cfg["max_k"]):
        try:
            p = Params(float(e), float(t), int(lo), int(hi)); sel_obj.check(p); grid.append(p)
        except ValueError:
            continue
    return grid, {"epsilon_candidates": eps, "tau_candidates": taus,
                  "score_quantiles_used": {"eps": cfg["epsilon_quantiles"], "tau": cfg["tau_quantiles"]}}


def run_grid(sel_obj, tasks, A, S, grid) -> pd.DataFrame:
    rows = []
    for p in grid:
        m = offline_metrics(sel_obj, tasks, A, S, p)
        rows.append({**p.asdict(), **m})
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated on (maximise write_cov, maximise complete_domain, minimise def_tokens)."""
    keep = []
    v = df[["write_cov", "complete_domain", "def_tokens"]].to_numpy()
    for i, a in enumerate(v):
        dom = any((b[0] >= a[0] and b[1] >= a[1] and b[2] <= a[2]) and (b[0] > a[0] or b[1] > a[1] or b[2] < a[2])
                  for j, b in enumerate(v) if j != i)
        keep.append(not dom)
    return df[keep]


def shortlist(grid_df: pd.DataFrame, n: int) -> pd.DataFrame:
    pf = pareto_front(grid_df).drop_duplicates(subset=["write_cov", "complete_domain", "def_tokens"])
    return pf.sort_values(["write_cov", "complete_domain", "def_tokens"], ascending=[False, False, True]).head(n)


def choose_final(val_runs: pd.DataFrame) -> dict:
    """val_runs: one row per (config_id, task) with success, total_tokens, retrieval_latency."""
    agg = val_runs.groupby("config_id").agg(success=("success", "mean"), tokens=("total_tokens", "mean"),
                                            retr_lat=("retrieval_latency", "mean")).reset_index()
    agg = agg.sort_values(["success", "tokens", "retr_lat"], ascending=[False, True, True])
    return agg
