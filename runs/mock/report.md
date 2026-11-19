> **MOCK RUN – simulated agent, synthetic data. Do not report.**

# Held-out results

Tasks: 550; templates: 55; excluded from outcome metrics due to infra errors: 0.

## Overall

| Metric | Mean [95% CI, template bootstrap] |
|---|---|
| Task success rate | 70.4% [63.1, 77.6] (n=550) |
| Domain recall | 96.7% [93.7, 99.1] (n=550) |
| Complete-domain coverage | 91.6% [84.2, 97.6] (n=550) |
| Write-tool coverage | 81.3% [71.6, 90.4] (n=390) |
| Complete write-tool coverage | 71.0% [57.7, 83.8] (n=390) |
| Side-effect rate | 0.0% [0.0, 0.0] (n=550) |
| |S(q)| | 6.52 [6.06, 6.94] (n=550) |
| Tool-definition tokens | 1.35e+03 [1.26e+03, 1.44e+03] (n=550) |
| Definition-token ratio vs full catalogue | 0.236 [0.22, 0.251] (n=550) |
| Input tokens | 791 [764, 816] (n=550) |
| Output tokens | 80 [80, 80] (n=550) |
| Total tokens | 871 [844, 896] (n=550) |
| Retrieval latency (s) | 6.72e-05 [6.37e-05, 7.11e-05] (n=550) |
| Total latency (s) | 2.02 [1.97, 2.07] (n=550) |


## multi-domain (n=170, templates=17)

| Metric | Mean [95% CI, template bootstrap] |
|---|---|
| Task success rate | 50.6% [36.5, 64.1] (n=170) |
| Domain recall | 89.3% [81.2, 96.3] (n=170) |
| Complete-domain coverage | 72.9% [52.3, 90.6] (n=170) |
| Write-tool coverage | 72.9% [59.4, 85.3] (n=170) |
| Complete write-tool coverage | 49.4% [27.6, 70.6] (n=170) |
| Side-effect rate | 0.0% [0.0, 0.0] (n=170) |
| |S(q)| | 7.83 [7.66, 7.98] (n=170) |
| Tool-definition tokens | 1.64e+03 [1.58e+03, 1.69e+03] (n=170) |
| Definition-token ratio vs full catalogue | 0.287 [0.276, 0.296] (n=170) |
| Input tokens | 870 [860, 879] (n=170) |
| Output tokens | 80 [80, 80] (n=170) |
| Total tokens | 950 [940, 959] (n=170) |
| Retrieval latency (s) | 6.24e-05 [5.92e-05, 6.59e-05] (n=170) |
| Total latency (s) | 2.04 [1.94, 2.13] (n=170) |

## single-domain (n=380, templates=38)

| Metric | Mean [95% CI, template bootstrap] |
|---|---|
| Task success rate | 79.2% [72.1, 85.8] (n=380) |
| Domain recall | 100.0% [100.0, 100.0] (n=380) |
| Complete-domain coverage | 100.0% [100.0, 100.0] (n=380) |
| Write-tool coverage | 87.7% [74.1, 100.0] (n=220) |
| Complete write-tool coverage | 87.7% [74.1, 100.0] (n=220) |
| Side-effect rate | 0.0% [0.0, 0.0] (n=380) |
| |S(q)| | 5.93 [5.36, 6.48] (n=380) |
| Tool-definition tokens | 1.22e+03 [1.11e+03, 1.32e+03] (n=380) |
| Definition-token ratio vs full catalogue | 0.214 [0.195, 0.231] (n=380) |
| Input tokens | 756 [722, 789] (n=380) |
| Output tokens | 80 [80, 80] (n=380) |
| Total tokens | 836 [802, 869] (n=380) |
| Retrieval latency (s) | 6.94e-05 [6.44e-05, 7.46e-05] (n=380) |
| Total latency (s) | 2.01 [1.96, 2.07] (n=380) |

## By number of domains

| #domains | n | success | complete-domain | mean |S(q)| |
|---|---|---|---|---|
| 1 | 380 | 79.2% | 100.0% | 5.9 |
| 2 | 120 | 57.5% | 83.3% | 7.8 |
| 3 | 30 | 33.3% | 33.3% | 8.0 |
| 4 | 20 | 35.0% | 70.0% | 8.0 |

## Hypotheses (association only, no causal claims)

**H1** complete-domain coverage, single minus multi-domain: 27.1 pp [9.5, 47.9] (single 100.0%, multi 72.9%). Supported if the CI excludes 0 and the difference is positive.
**H2** success, complete minus incomplete write-tool coverage: 66.8 pp [60.4, 73.1] (complete n=277, incomplete n=113).
**H3** Spearman rho(#domains, |S(q)|) = 0.566 [0.420, 0.685].

## Failure analysis

| Category | n | % of all evaluated |
|---|---|---|
| infra_error | 550 | 100.0% |

Categories are descriptive. 'execution_error_or_read_tool_omission' cannot be separated without inspecting traces (ground truth lists only write actions); sample these manually.

Selector flags: truncated_to_max_k=213