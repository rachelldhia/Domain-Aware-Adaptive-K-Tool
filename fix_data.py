p = "adak/data.py"
s = open(p, encoding="utf-8").read()
a = s.index("    # 2026 repo layout")
b = s.index("    tasks, i = [], 0")
new = (
    '    files = sorted(glob.glob(os.path.join(repo, "data", "processed", "tasks_and_outcomes", "*_tasks_and_outcomes.csv")))\n'
    '    assert len(files) == 6, f"expected 6 top-level task files (v2), got {len(files)}: {files}"\n'
)
open(p, "w", encoding="utf-8").write(s[:a] + new + s[b:])
print("data.py diperbaiki")