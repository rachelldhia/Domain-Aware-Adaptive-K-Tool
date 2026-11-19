"""Held-out analysis (Section 3.2.8): template-level bootstrap CIs, H1-H3, failure coding."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def failure_category(r) -> str:
    if r["infra_error"]: return "infra_error"
    if r["success"]: return "success"
    d_om, t_om = not r["complete_domain"], bool(r["write_tool_omitted_in_selected_domain"])
    if d_om and t_om: return "domain+tool_omission"
    if d_om: return "domain_omission"
    if t_om: return "tool_omission_in_selected_domain"
    return "execution_error_or_read_tool_omission"      # read-tool needs are not observable from ground truth


def boot(df: pd.DataFrame, stat, B: int, seed: int = 0):
    """Cluster bootstrap: resample base_template groups (10 tasks share a template)."""
    groups = [g.index.to_numpy() for _, g in df.groupby("base_template")]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(df.loc[np.concatenate([groups[i] for i in pick])]))
    vals = np.array(vals, dtype=float)
    return np.nanpercentile(vals, [2.5, 97.5]) if np.isfinite(vals).any() else (np.nan, np.nan)


def mean_ci(df, col, B):
    ok = df[df[col].notna()]
    if ok.empty: return (np.nan, np.nan, np.nan, 0)
    lo, hi = boot(ok.reset_index(drop=True), lambda d: d[col].mean(), B)
    return ok[col].mean(), lo, hi, len(ok)


def fmt(m, lo, hi, n=None, pct=False):
    if np.isnan(m): return "n/a"
    k = 100 if pct else 1
    s = f"{m*k:.1f}% [{lo*k:.1f}, {hi*k:.1f}]" if pct else f"{m:.3g} [{lo:.3g}, {hi:.3g}]"
    return s + (f" (n={n})" if n is not None else "")


METRICS = [("Task success rate", "success", True), ("Domain recall", "domain_recall", True),
           ("Complete-domain coverage", "complete_domain", True), ("Write-tool coverage", "write_cov", True),
           ("Complete write-tool coverage", "complete_write", True), ("Side-effect rate", "side_effects", True),
           ("|S(q)|", "n_tools", False), ("Tool-definition tokens", "def_tokens", False),
           ("Definition-token ratio vs full catalogue", "token_ratio", False),
           ("Input tokens", "input_tokens", False), ("Output tokens", "output_tokens", False),
           ("Total tokens", "total_tokens", False), ("Retrieval latency (s)", "retrieval_latency", False),
           ("Total latency (s)", "latency", False)]


def metric_table(df, B) -> str:
    rows = ["| Metric | Mean [95% CI, template bootstrap] |", "|---|---|"]
    for name, col, pct in METRICS:
        if col not in df: continue
        m, lo, hi, n = mean_ci(df.assign(**{col: df[col].astype(float)}), col, B)
        rows.append(f"| {name} | {fmt(m, lo, hi, n, pct)} |")
    return "\n".join(rows)


def analyse(logs: pd.DataFrame, B: int = 2000) -> str:
    df = logs.reset_index(drop=True).copy()
    df["failure_category"] = df.apply(failure_category, axis=1)
    n_infra = int(df["infra_error"].notna().sum())
    ev = df[df["infra_error"].isna()].reset_index(drop=True)          # task outcomes exclude infra errors
    out = [f"# Held-out results\n\nTasks: {len(df)}; templates: {df.base_template.nunique()}; "
           f"excluded from outcome metrics due to infra errors: {n_infra}.\n",
           "## Overall\n", metric_table(ev, B), ""]
    ev["group"] = np.where(ev.n_domains == 1, "single-domain", "multi-domain")
    for g, d in ev.groupby("group"):
        out += [f"\n## {g} (n={len(d)}, templates={d.base_template.nunique()})\n", metric_table(d.reset_index(drop=True), B)]
    out += ["\n## By number of domains\n", "| #domains | n | success | complete-domain | mean |S(q)| |", "|---|---|---|---|---|"]
    for k, d in ev.groupby("n_domains"):
        out.append(f"| {k} | {len(d)} | {d.success.mean()*100:.1f}% | {d.complete_domain.mean()*100:.1f}% | {d.n_tools.mean():.1f} |")

    # ---------------- hypotheses
    out.append("\n## Hypotheses (association only, no causal claims)\n")
    # H1
    def h1(d):
        s, m = d[d.n_domains == 1].complete_domain.mean(), d[d.n_domains > 1].complete_domain.mean()
        return s - m
    d1 = h1(ev); lo, hi = boot(ev, h1, B)
    out.append(f"**H1** complete-domain coverage, single minus multi-domain: {d1*100:.1f} pp [{lo*100:.1f}, {hi*100:.1f}] "
               f"(single {ev[ev.n_domains==1].complete_domain.mean()*100:.1f}%, multi {ev[ev.n_domains>1].complete_domain.mean()*100:.1f}%). "
               f"Supported if the CI excludes 0 and the difference is positive.")
    # H2
    w = ev[ev.complete_write.notna()].reset_index(drop=True)
    def h2(d):
        a, b = d[d.complete_write == 1].success, d[d.complete_write == 0].success
        return a.mean() - b.mean() if len(a) and len(b) else np.nan
    n_c, n_i = int((w.complete_write == 1).sum()), int((w.complete_write == 0).sum())
    d2 = h2(w); lo, hi = boot(w, h2, B)
    out.append(f"**H2** success, complete minus incomplete write-tool coverage: "
               f"{d2*100:.1f} pp [{lo*100:.1f}, {hi*100:.1f}] (complete n={n_c}, incomplete n={n_i})."
               + (" Incomplete group is very small; interpret with caution." if n_i < 20 else ""))
    # H3
    rho = spearmanr(ev.n_domains, ev.n_tools)[0]
    lo, hi = boot(ev, lambda d: spearmanr(d.n_domains, d.n_tools)[0], B)
    out.append(f"**H3** Spearman rho(#domains, |S(q)|) = {rho:.3f} [{lo:.3f}, {hi:.3f}].")

    out += ["\n## Failure analysis\n", "| Category | n | % of all evaluated |", "|---|---|---|"]
    vc = ev.assign(fc=ev.apply(failure_category, axis=1)).fc.value_counts()
    for k, v in vc.items(): out.append(f"| {k} | {v} | {v/len(ev)*100:.1f}% |")
    out.append("\nCategories are descriptive. 'execution_error_or_read_tool_omission' cannot be separated without inspecting traces "
               "(ground truth lists only write actions); sample these manually.")
    flags = df["flags"].explode().value_counts() if "flags" in df else pd.Series(dtype=int)
    if len(flags): out += ["\nSelector flags: " + ", ".join(f"{k}={v}" for k, v in flags.items())]
    return "\n".join(out), df
