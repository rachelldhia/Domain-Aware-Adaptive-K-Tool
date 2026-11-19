#!/usr/bin/env python3
"""Domain-Aware Adaptive-k on WorkBench.

Stages (run in order; every stage writes to --out):
  prepare   load data, build tool embeddings, template-level split
  tune      offline grid on VALIDATION tasks -> shortlist
  validate  run agent on shortlist (validation tasks) -> freeze best config
  test      run frozen config once on held-out tasks (refuses without frozen_config.json)
  report    CIs, H1-H3, failure coding
Add --mock for a synthetic dry run (no network, no LLM; outputs are NOT results).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
from adak import data as D
from adak.encoders import make_encoder
from adak.selector import DomainAwareAdaptiveK, Params
from adak.split import template_split, apply_split
from adak import tune as T
from adak.agent import MockAgent
from adak.tune import task_row
from adak.evaluate import analyse


def load(cfg, mock):
    if mock:
        tools, tasks = D.mock_tools(cfg["tool_text"]["split_name"]), D.mock_tasks(cfg["seed"])
    else:
        tools = D.load_workbench_tools(cfg["workbench_repo"], cfg["tool_text"]["split_name"])
        tasks = D.load_workbench_tasks(cfg["workbench_repo"])
    enc = make_encoder(cfg["encoder"], mock)
    return tools, tasks, enc, DomainAwareAdaptiveK(tools, enc)


def make_agent(cfg, mock, tools):
    if mock: return MockAgent(cfg["seed"])
    from adak.agent import RealAgent
    return RealAgent(cfg["agent"], {t.id: t for t in tools}, cfg["workbench_repo"])


def jdump(o, p): json.dump(o, open(p, "w"), indent=2, default=str)


def run_tasks(selector, agent, tasks, params: Params, tools_by_id, config_id=None):
    t0 = time.perf_counter()
    A, S = selector.score([t.query for t in tasks])           # per-query encoding cost is timed below
    rows = []
    for i, t in enumerate(tasks):
        r0 = time.perf_counter()
        a, s = selector.score([t.query]); sel = selector.select(a[0], s[0], params)
        retr = time.perf_counter() - r0
        rec = agent.run(t, sel.tool_ids)
        row = task_row(selector, t, sel)
        miss_in_dom = [w for w in t.write_tools if w not in sel.tool_ids and tools_by_id[w].domain in sel.domains]
        rows.append({"config_id": config_id, "task_idx": t.idx, "query": t.query, "base_template": t.base_template,
                     "n_domains": t.n_domains, "true_domains": t.domains, "pred_domains": sel.domains,
                     "write_tools": t.write_tools, "selected_tools": sel.tool_ids, "flags": sel.flags,
                     "write_tool_omitted_in_selected_domain": bool(miss_in_dom),
                     "token_ratio": row["def_tokens"] / selector.full_catalogue_tokens,
                     "retrieval_latency": retr, **row, **{k: v for k, v in vars(rec).items()}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "tune", "validate", "test", "report"])
    ap.add_argument("--config", default="config.json"); ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config)); os.makedirs(a.out, exist_ok=True)
    P = lambda f: os.path.join(a.out, f)
    tools, tasks, enc, selector = load(cfg, a.mock)
    tools_by_id = {t.id: t for t in tools}
    if a.mock: print("*** MOCK MODE: synthetic data, hashing encoder, simulated agent. Not experimental results. ***")

    if a.stage == "prepare":
        sp = template_split(tasks, cfg["split"]["n_validation_templates"], cfg["seed"])
        val, test = apply_split(tasks, sp)
        sp.update({"n_val_tasks": len(val), "n_test_tasks": len(test), "encoder": enc.name,
                   "workbench_commit": cfg["workbench_commit"], "n_tools": len(tools),
                   "full_catalogue_tokens": selector.full_catalogue_tokens, "mock": a.mock})
        jdump(sp, P("split.json"))
        print(f"validation: {len(val)} tasks / {len(sp['validation_templates'])} templates; test: {len(test)} / {len(sp['test_templates'])}")

    elif a.stage == "tune":
        val, _ = apply_split(tasks, json.load(open(P("split.json"))))
        A, S = selector.score([t.query for t in val])
        grid, meta = T.build_grid(selector, A, S, cfg["grid"])
        df = T.run_grid(selector, val, A, S, grid)
        df.to_csv(P("grid_offline_validation.csv"), index=False)
        sl = T.shortlist(df, cfg["shortlist_size"]).reset_index(drop=True); sl["config_id"] = [f"c{i}" for i in range(len(sl))]
        sl.to_csv(P("shortlist.csv"), index=False); jdump(meta, P("grid_meta.json"))
        print(f"{len(grid)} configs evaluated; shortlist:\n{sl.round(3).to_string()}")

    elif a.stage == "validate":
        val, _ = apply_split(tasks, json.load(open(P("split.json"))))
        sl = pd.read_csv(P("shortlist.csv")); agent = make_agent(cfg, a.mock, tools); rows = []
        for _, c in sl.iterrows():
            p = Params(float(c.epsilon), float(c.tau), int(c.min_k), int(c.max_k))
            rows += run_tasks(selector, agent, val, p, tools_by_id, c.config_id)
        vr = pd.DataFrame(rows); vr.to_json(P("validation_agent_runs.jsonl"), orient="records", lines=True)
        ok = vr[vr.infra_error.isna()]
        ranking = T.choose_final(ok); ranking.to_csv(P("validation_ranking.csv"), index=False)
        best = sl[sl.config_id == ranking.iloc[0].config_id].iloc[0]
        frozen = {"params": Params(float(best.epsilon), float(best.tau), int(best.min_k), int(best.max_k)).asdict(),
                  "config_id": best.config_id, "validation_success": float(ranking.iloc[0].success),
                  "encoder": enc.name, "agent": cfg["agent"]["model"], "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "mock": a.mock}
        frozen["sha256"] = hashlib.sha256(json.dumps(frozen["params"], sort_keys=True).encode()).hexdigest()
        jdump(frozen, P("frozen_config.json")); print(ranking.round(3).to_string()); print("FROZEN:", frozen["params"])

    elif a.stage == "test":
        if not os.path.exists(P("frozen_config.json")): sys.exit("Refusing: no frozen_config.json (run `validate` first).")
        if os.path.exists(P("test_logs.jsonl")): sys.exit("Refusing: test already run. Test set is evaluated once; delete manually only for a documented repeat.")
        fz = json.load(open(P("frozen_config.json"))); p = Params(**fz["params"])
        _, test = apply_split(tasks, json.load(open(P("split.json"))))
        rows = run_tasks(selector, make_agent(cfg, a.mock, tools), test, p, tools_by_id, fz["config_id"])
        pd.DataFrame(rows).to_json(P("test_logs.jsonl"), orient="records", lines=True)
        print(f"{len(rows)} test tasks logged")

    elif a.stage == "report":
        logs = pd.read_json(P("test_logs.jsonl"), lines=True)
        text, df = analyse(logs, cfg["bootstrap"]["B"])
        if a.mock: text = "> **MOCK RUN – simulated agent, synthetic data. Do not report.**\n\n" + text
        open(P("report.md"), "w").write(text); df.to_csv(P("test_task_level.csv"), index=False); print(text)


if __name__ == "__main__":
    main()
