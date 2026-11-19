"""Agent execution, logging, and state-based evaluation.

REAL MODE depends on WorkBench internals (src.tools.*). The adapter layer is isolated in
`WorkBenchSandbox`; verify its assumptions against the pinned commit before running.
"""
from __future__ import annotations
import json, os, random, time
from dataclasses import dataclass, field, asdict


@dataclass
class RunRecord:
    predicted_calls: list = field(default_factory=list)
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0
    latency: float = 0.0
    steps: int = 0
    success: bool = False
    side_effects: bool = False
    infra_error: str | None = None        # API failure / timeout / evaluator error (NOT a task failure)
    agent_error: str | None = None


# ------------------------------------------------------------------ real sandbox
class WorkBenchSandbox:
    MODULES = ["email", "calendar", "analytics", "project_management",
               "customer_relationship_manager", "company_directory"]

    def __init__(self, repo: str):
        import sys, importlib
        sys.path.insert(0, os.path.abspath(repo))
        self.mods = {m: importlib.import_module(f"src.tools.{m}") for m in self.MODULES}

    def reset(self):
        for m in self.mods.values():
            if hasattr(m, "reset_state"): m.reset_state()

    def snapshot(self):
        import pandas as pd
        snap = {}
        for n, m in self.mods.items():
            for k, v in vars(m).items():
                if isinstance(v, pd.DataFrame): snap[f"{n}.{k}"] = v.copy(deep=True)
        return snap

    def execute(self, call: str):
        return eval(call, {**self.mods})      # WorkBench answers are strings like "email.send_email.func(...)"

    @staticmethod
    def same(a, b):
        return a.keys() == b.keys() and all(a[k].equals(b[k]) for k in a)

    def judge(self, predicted_calls, gt_calls):
        """Final-state comparison. Prefer WorkBench's own evaluator; fall back to snapshot equality."""
        try:
            from src.evals import utils as wb
            if hasattr(wb, "is_correct") and hasattr(wb, "has_side_effects"):
                # signature may differ between commits -> adapt here
                return bool(wb.is_correct(predicted_calls, gt_calls, None)), bool(wb.has_side_effects(predicted_calls, gt_calls))
        except Exception:
            pass
        self.reset()
        for c in gt_calls: self.execute(c)
        gt_state = self.snapshot()
        self.reset()
        for c in predicted_calls:
            try: self.execute(c)
            except Exception: pass
        ok = self.same(self.snapshot(), gt_state)
        self.reset()
        return ok, False


def _call_str(tool_name: str, args: dict) -> str:
    return f"{tool_name}.func({', '.join(f'{k}={v!r}' for k, v in args.items())})"


class RealAgent:
    """OpenAI-compatible tool-calling loop restricted to S(q)."""
    def __init__(self, cfg: dict, tools_by_id: dict, repo: str):
        from openai import OpenAI
        from langchain_core.utils.function_calling import convert_to_openai_function
        self.cfg, self.tools = cfg, tools_by_id
        self.client = OpenAI(base_url=cfg.get("base_url"), api_key=os.environ[cfg["api_key_env"]],
                             timeout=cfg["timeout_s"])
        self.fn_schema = {i: {"type": "function", "function": {**convert_to_openai_function(t.obj), "name": i}}
                          for i, t in tools_by_id.items()}
        self.sb = WorkBenchSandbox(repo)

    def run(self, task, tool_ids) -> RunRecord:
        rec, t0 = RunRecord(), time.perf_counter()
        self.sb.reset()
        msgs = [{"role": "system", "content": self.cfg["system_prompt"]}, {"role": "user", "content": task.query}]
        tools = [self.fn_schema[i] for i in tool_ids]
        try:
            for step in range(self.cfg["max_steps"]):
                r = self.client.chat.completions.create(model=self.cfg["model"], messages=msgs, tools=tools,
                                                        temperature=self.cfg["temperature"])
                rec.steps += 1
                if r.usage:
                    rec.input_tokens += r.usage.prompt_tokens; rec.output_tokens += r.usage.completion_tokens
                else:
                    rec.input_tokens = rec.output_tokens = None      # report limitation, never estimate
                m = r.choices[0].message
                msgs.append(m.model_dump(exclude_none=True))
                if not m.tool_calls: break
                for tc in m.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    tool = self.tools.get(tc.function.name)
                    call = _call_str(tool.obj.name, args) if tool else f"UNKNOWN.{tc.function.name}"
                    rec.predicted_calls.append(call)
                    try: out = str(self.sb.execute(call))
                    except Exception as e: out = f"Error: {e}"; rec.agent_error = rec.agent_error or str(e)[:200]
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        except Exception as e:
            rec.infra_error = f"{type(e).__name__}: {str(e)[:200]}"
        rec.latency = time.perf_counter() - t0
        if rec.input_tokens is not None: rec.total_tokens = rec.input_tokens + rec.output_tokens
        else: rec.total_tokens = None
        if not rec.infra_error:
            try: rec.success, rec.side_effects = self.sb.judge(rec.predicted_calls, task.gt_calls)
            except Exception as e: rec.infra_error = f"evaluator: {e}"
        return rec


# ------------------------------------------------------------------ mock agent
class MockAgent:
    """SIMULATED agent for pipeline testing only. Outcomes are random draws and MUST NOT be reported."""
    def __init__(self, seed=0): self.rng = random.Random(seed)

    def run(self, task, tool_ids) -> RunRecord:
        have = set(tool_ids); need = set(task.write_tools)
        p = 0.85 if need <= have else 0.15
        n_in = 400 + 60 * len(tool_ids)
        return RunRecord(predicted_calls=[], input_tokens=n_in, output_tokens=80, total_tokens=n_in + 80,
                         latency=self.rng.uniform(1, 3), steps=2, success=self.rng.random() < p)
