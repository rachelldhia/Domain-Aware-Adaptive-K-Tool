"""Jalankan dari folder adak_experiment:  python inspect_workbench.py ./WorkBench
Mencetak struktur repo WorkBench supaya adapter bisa disesuaikan. Salin seluruh output ke chat."""
import os, sys, glob, re
repo = sys.argv[1] if len(sys.argv) > 1 else "./WorkBench"
print("== commit ==")
head = os.path.join(repo, ".git", "HEAD")
print(open(head).read().strip() if os.path.exists(head) else "(tidak ada .git)")
print("\n== src tree ==")
for r, d, f in os.walk(os.path.join(repo, "src")):
    d[:] = [x for x in d if x not in ("__pycache__",)]
    for x in sorted(f):
        if x.endswith(".py"): print(os.path.relpath(os.path.join(r, x), repo))
print("\n== task files ==")
try:
    import pandas as pd
    for f in sorted(glob.glob(os.path.join(repo, "data", "processed", "tasks_and_outcomes", "*.csv"))):
        df = pd.read_csv(f); print(os.path.relpath(f, repo), df.shape, list(df.columns))
except Exception as e:
    print("pandas error:", e)
print("\n== kata kunci di src ==")
pat = re.compile(r"^(def run_agent\w*|MODEL_REGISTRY|class \w+|\w+_toolkit\w*\s*=|def reset_state|def \w*tool\w*)", re.M)
for r, d, f in os.walk(os.path.join(repo, "src")):
    for x in sorted(f):
        if x.endswith(".py"):
            p = os.path.join(r, x); hits = pat.findall(open(p, encoding="utf-8", errors="ignore").read())
            if hits: print(os.path.relpath(p, repo), "->", sorted(set(hits))[:25])
