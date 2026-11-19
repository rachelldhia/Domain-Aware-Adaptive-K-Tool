"""Tool catalogue + task loading (real WorkBench adapter and synthetic mock)."""
from __future__ import annotations
import ast, glob, json, os, re, sys
from dataclasses import dataclass, field
import numpy as np

SHARED = "company_directory"
DOMAINS = ["analytics", "calendar", "customer_relationship_manager", "email", "project_management"]
ALIASES = {
    "analytics": "analytics", "web analytics": "analytics", "web_analytics": "analytics",
    "calendar": "calendar", "crm": "customer_relationship_manager",
    "customer_relationship_manager": "customer_relationship_manager",
    "customer relationship manager": "customer_relationship_manager",
    "email": "email", "project management": "project_management",
    "project_management": "project_management",
}


def norm_id(name: str) -> str:
    """'email.send_email' / 'email_send_email' -> 'email_send_email'."""
    return name.strip().lower().replace(".", "_")


def norm_domain(d: str) -> str:
    return ALIASES[d.strip().lower().replace("-", "_")] if d.strip().lower().replace("-", "_") in ALIASES else ALIASES[d.strip().lower()]


@dataclass
class Tool:
    id: str                 # normalised id
    domain: str | None      # None for shared utility
    text: str               # string that gets embedded
    def_tokens: int         # tokens of serialized definition
    obj: object = field(default=None, repr=False)   # WorkBench/langchain tool (real mode)


@dataclass
class Task:
    idx: int
    query: str
    base_template: str
    domains: list[str]
    gt_calls: list[str]
    write_tools: list[str]   # normalised ids

    @property
    def n_domains(self): return len(self.domains)


_CALL_RE = re.compile(r"^\s*([A-Za-z_]+\.[A-Za-z_]+)\.func\(")


def write_tools_from_calls(calls: list[str]) -> list[str]:
    out = []
    for c in calls:
        m = _CALL_RE.match(c)
        if m and norm_id(m.group(1)) not in out:
            out.append(norm_id(m.group(1)))
    return out


def count_tokens(s: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(s))
    except Exception:
        return max(1, round(len(s) / 4))   # fallback, flagged in README


def tool_text(name: str, desc: str, params: dict, split_name: bool = True) -> str:
    """name + description (docstring) + parameter names. In the 2026 WorkBench `Tool`, args_schema only has
    title/type/default per parameter (no separate description), so parameter 'descriptions' = names + types."""
    n = name.replace("_", " ").replace(".", " ") if split_name else name
    ptxt = "; ".join(f"{k} ({(v or {}).get('type', 'string')})" for k, v in params.items())
    return f"{n}. {' '.join((desc or '').split())} Parameters: {ptxt}".strip()


# ---------------------------------------------------------------- real WorkBench
DOMAIN_MODULES = {   # module under src/tools -> domain (company_directory = shared set U)
    "analytics": "analytics", "calendar": "calendar", "email": "email",
    "customer_relationship_manager": "customer_relationship_manager",
    "project_management": "project_management", SHARED: None,
}


def load_workbench_tools(repo: str, split_name: bool = True, token_render: str = "render_tool_description") -> list[Tool]:
    """Collect every `Tool` instance defined in src/tools/<domain>.py (WorkBench 2026 layout).
    token_render: 'render_tool_description' (text prompt, ReAct) or 'openai_schema' (structured tool calling)."""
    import importlib
    sys.path.insert(0, os.path.abspath(repo))
    from src.tools.tool import Tool as WBTool, render_tool_description, tool_to_openai_schema
    tools, seen = [], set()
    for mod_name, dom in DOMAIN_MODULES.items():
        mod = importlib.import_module(f"src.tools.{mod_name}")
        found = sorted((v for v in vars(mod).values() if isinstance(v, WBTool)), key=lambda t: t.name)
        for t in found:
            if t.name in seen: continue
            seen.add(t.name)
            ser = render_tool_description(t) if token_render == "render_tool_description" else json.dumps(tool_to_openai_schema(t))
            tools.append(Tool(norm_id(t.name), dom, tool_text(t.name, t.description, t.args_schema, split_name),
                              count_tokens(ser), t))
    print(f"[load_workbench_tools] {len(tools)} tools: " +
          ", ".join(f"{d or 'shared'}={sum(1 for t in tools if t.domain == d)}" for d in DOMAIN_MODULES.values()))
    if len(tools) != 27:
        print("WARNING: expected 27 tools (26 domain + company_directory). Compare with toolkits.py before trusting results.")
    return tools


def _parse_list(x):
    if isinstance(x, list): return x
    if x is None or (isinstance(x, float) and np.isnan(x)): return []
    s = str(x).strip()
    if not s: return []
    try:
        v = ast.literal_eval(s)
        return list(v) if isinstance(v, (list, tuple)) else [v]
    except Exception:
        return [p for p in re.split(r"[;,]", s) if p.strip()]


def load_workbench_tasks(repo: str, col_query="query", col_answer="answer") -> list[Task]:
    import pandas as pd
    # 2026 repo layout: data/processed/tasks_and_outcomes/*_tasks_and_outcomes.csv (v2 = top-level; v1 = old snapshot)
    files = sorted(glob.glob(os.path.join(repo, "data", "processed", "tasks_and_outcomes", "*_tasks_and_outcomes.csv")))
    assert len(files) == 6, f"expected 6 top-level task files (v2), got {len(files)}: {files}"
    tasks, i = [], 0
    for f in files:
        df = pd.read_csv(f)
        multi = "multi" in os.path.basename(f).lower()
        file_dom = None
        if not multi:
            b = os.path.basename(f).lower()
            for key, d in [("customer", "customer_relationship_manager"), ("crm", "customer_relationship_manager"),
                           ("project", "project_management"), ("analytics", "analytics"),
                           ("calendar", "calendar"), ("email", "email")]:
                if key in b: file_dom = d; break
        qc = col_query if col_query in df else "task"
        ac = col_answer if col_answer in df else "outcome"
        for _, r in df.iterrows():
            doms = ([norm_domain(d) for d in _parse_list(r["domains"])] if multi and "domains" in df
                    else [file_dom])
            calls = [str(c) for c in _parse_list(r[ac])]
            tasks.append(Task(i, str(r[qc]), str(r.get("base_template", f"{os.path.basename(f)}:{i//10}")),
                              doms, calls, write_tools_from_calls(calls)))
            i += 1
    return tasks


# ---------------------------------------------------------------- mock data
_MOCK_TOOLS = {
 "email": [("get_email_information_by_id", "Retrieve details of an email by its id.", "email_id: id of the email"),
           ("search_emails", "Search emails by query, sender or date range.", "query: text to look for; date_min: earliest date; date_max: latest date"),
           ("send_email", "Send an email message to a recipient.", "recipient: email address; subject: subject line; body: message body"),
           ("delete_email", "Delete an email from the inbox.", "email_id: id of the email"),
           ("forward_email", "Forward an existing email to another recipient.", "email_id: id of the email; recipient: email address"),
           ("reply_email", "Reply to an existing email with a message.", "email_id: id of the email; body: reply body")],
 "calendar": [("get_event_information_by_id", "Get details of a calendar event by id.", "event_id: id of the event"),
              ("search_events", "Search calendar events by name or time range.", "query: event name; time_min: start of range; time_max: end of range"),
              ("create_event", "Create a new calendar event or meeting with participants.", "event_name: title; participant_email: attendee; event_start: start time; duration: minutes"),
              ("delete_event", "Delete a calendar event.", "event_id: id of the event"),
              ("update_event", "Update or reschedule a calendar event field such as time or title.", "event_id: id; field: field to change; new_value: new value")],
 "analytics": [("get_visitor_information_by_id", "Get website visitor information by visitor id.", "visitor_id: id of the visitor"),
               ("create_plot", "Plot website analytics such as visits over a time range.", "time_min: start date; time_max: end date; value_to_plot: metric"),
               ("total_visits_count", "Count total website visits in a date range.", "time_min: start date; time_max: end date"),
               ("engaged_users_count", "Count engaged website users in a date range.", "time_min: start date; time_max: end date"),
               ("traffic_source_count", "Count website traffic from a source such as search or social.", "time_min: start date; time_max: end date; traffic_source: source"),
               ("get_average_session_duration", "Average website session duration in a date range.", "time_min: start date; time_max: end date")],
 "customer_relationship_manager": [
               ("search_customers", "Search CRM customers by name, status, product interest or assigned sales rep.", "customer_name: name; customer_status: lead or won; product_interest: product"),
               ("update_customer", "Update a CRM customer record field such as status or notes.", "customer_id: id; field: field to change; new_value: new value"),
               ("add_customer", "Add a new customer to the CRM.", "customer_name: name; assigned_to_email: sales rep; customer_status: status"),
               ("delete_customer", "Delete a customer from the CRM.", "customer_id: id")],
 "project_management": [
               ("get_task_information_by_id", "Get details of a project task by id.", "task_id: id of the task"),
               ("search_tasks", "Search project tasks by name, assignee, board or status.", "task_name: name; assigned_to_email: assignee; list_name: board; due_date: date"),
               ("create_task", "Create a new project task on a board.", "task_name: name; assigned_to_email: assignee; list_name: board; due_date: due date"),
               ("delete_task", "Delete a project task.", "task_id: id"),
               ("update_task", "Update a project task field such as board, status or due date.", "task_id: id; field: field to change; new_value: new value")],
 SHARED: [("find_email_address", "Find an email address from a person's name in the company directory.", "name: name of the person")],
}
_WRITE = {"send_email", "delete_email", "forward_email", "reply_email", "create_event", "delete_event", "update_event",
          "update_customer", "add_customer", "delete_customer", "create_task", "delete_task", "update_task"}


def mock_tools(split_name: bool = True) -> list[Tool]:
    out = []
    for dom, items in _MOCK_TOOLS.items():
        for n, d, p in items:
            params = {kv.split(":")[0].strip(): {"description": kv.split(":", 1)[1].strip()} for kv in p.split(";")}
            full = f"{dom}.{n}" if dom != SHARED else f"{SHARED}.{n}"
            text = tool_text(full, d, params, split_name)
            out.append(Tool(norm_id(full), None if dom == SHARED else dom, text, count_tokens(text) * 6))
    return out


_SINGLE = {
 "email": [("Send an email to {p} with subject {s} about the quarterly report", ["send_email"]),
           ("Delete the email from {p} about {s}", ["delete_email"]),
           ("Forward the latest email about {s} to {p}", ["forward_email"]),
           ("Reply to {p}'s email about {s} saying thanks", ["reply_email"]),
           ("Find the last email from {p} about {s}", []),
           ("Send {p} a reminder email and then delete the old email about {s}", ["send_email", "delete_email"]),
           ("How many emails did I get from {p} about {s}", []),
           ("Reply to every email about {s} from {p}", ["reply_email"]),
           ("Delete all emails from {p} that mention {s}", ["delete_email"])],
 "calendar": [("Schedule a meeting with {p} called {s} tomorrow at 2pm", ["create_event"]),
              ("Cancel the {s} meeting with {p}", ["delete_event"]),
              ("Move the {s} meeting with {p} to Friday", ["update_event"]),
              ("What meetings do I have with {p} about {s}", []),
              ("Create a meeting {s} with {p} and delete the old event", ["create_event", "delete_event"]),
              ("Reschedule all {s} events to next week", ["update_event"]),
              ("When is my next event with {p} about {s}", []),
              ("Rename the event {s} with {p} to include the word planning", ["update_event"]),
              ("Delete every {s} event on Monday", ["delete_event"]),
              ("Set up a recurring {s} meeting with {p}", ["create_event"]),
              ("Show the details of the {s} event with {p}", [])],
 "analytics": [("How many website visits did we have last week from {s} traffic", []),
               ("Plot the number of engaged users from the start of November for {s}", []),
               ("What was the average session duration on the website for {s}", []),
               ("Plot total visits between Nov 1 and Nov 15 for {s} traffic", []),
               ("How many users came from search engines during {s}", []),
               ("Compare engaged users and total visits for {s}", []),
               ("Which day had the most visits for {s}", []),
               ("Show a chart of website traffic sources for {s}", []),
               ("What is the average session length for visitors from {s}", []),
               ("Count total visits for {s} last month", []),
               ("Plot engaged users in the last two weeks for {s}", []),
               ("How many visitors were engaged during {s} week", [])],
 "customer_relationship_manager": [
               ("Add {p} as a new customer interested in {s}", ["add_customer"]),
               ("Delete the customer {p} from the CRM", ["delete_customer"]),
               ("Change the status of customer {p} to won", ["update_customer"]),
               ("Which customers are interested in {s}", []),
               ("Update the notes of customer {p} with {s}", ["update_customer"]),
               ("Reassign all customers interested in {s} to {p}", ["update_customer"]),
               ("List leads assigned to {p} interested in {s}", []),
               ("Remove every lost customer interested in {s}", ["delete_customer"])],
 "project_management": [
               ("Create a task {s} for {p} on the backlog board due Friday", ["create_task"]),
               ("Delete the task {s}", ["delete_task"]),
               ("Move {p}'s task {s} to the done board", ["update_task"]),
               ("What tasks are assigned to {p} about {s}", []),
               ("Reassign the {s} task to {p}", ["update_task"]),
               ("Create a task {s} and delete the old {s} task", ["create_task", "delete_task"]),
               ("Show all overdue tasks for {p} about {s}", []),
               ("Change the due date of task {s} to next Monday", ["update_task"])],
}
_MULTI = [
 (("calendar", "email"), "Move the {s} meeting with {p} to Friday and notify them by email", ["update_event", "send_email"]),
 (("calendar", "email"), "Create a {s} event with {p} and send them a confirmation email", ["create_event", "send_email"]),
 (("calendar", "project_management"), "Schedule a {s} meeting with {p} and create a task {s} for them", ["create_event", "create_task"]),
 (("customer_relationship_manager", "email"), "Email {p} and update customer {p}'s status to won because of {s}", ["send_email", "update_customer"]),
 (("customer_relationship_manager", "project_management"), "Add {p} as a customer and create a task {s} for the sales rep", ["add_customer", "create_task"]),
 (("analytics", "email"), "Email {p} the number of visits last week from {s} traffic", ["send_email"]),
 (("analytics", "project_management"), "Create a task {s} for {p} if website visits dropped last week", ["create_task"]),
 (("analytics", "calendar"), "Schedule a meeting with {p} to review {s} traffic if engaged users dropped", ["create_event"]),
 (("email", "project_management"), "Delete the task {s} and send {p} an email saying it is done", ["delete_task", "send_email"]),
 (("calendar", "customer_relationship_manager"), "Set up a {s} meeting with {p} and mark customer {p} as won", ["create_event", "update_customer"]),
 (("customer_relationship_manager", "analytics"), "Update customers interested in {s} and plot visits for {s}", ["update_customer"]),
 (("calendar", "email", "project_management"), "Create a {s} event with {p}, email them, and create a task {s} for them", ["create_event", "send_email", "create_task"]),
 (("customer_relationship_manager", "email", "calendar"), "Add customer {p}, send them a welcome email and book a {s} meeting", ["add_customer", "send_email", "create_event"]),
 (("analytics", "email", "calendar"), "Email {p} the {s} visit count and schedule a review meeting", ["send_email", "create_event"]),
 (("project_management", "calendar", "email"), "Reassign the {s} task to {p}, move their meeting and email them", ["update_task", "update_event", "send_email"]),
 (("analytics", "customer_relationship_manager", "email", "project_management"), "For {s}, plot visits, update the customer {p}, email {p} and create a task", ["update_customer", "send_email", "create_task"]),
 (("analytics", "calendar", "email", "project_management"), "Plot {s} visits, schedule a meeting with {p}, email them and add a task", ["create_event", "send_email", "create_task"]),
 (("calendar", "email"), "Delete the {s} event with {p} and email them that it was cancelled", ["delete_event", "send_email"]),
 (("project_management", "email"), "Move the {s} task to done and email {p} about it", ["update_task", "send_email"]),
 (("customer_relationship_manager", "calendar"), "Delete customer {p} and cancel the {s} meeting with them", ["delete_customer", "delete_event"]),
 (("analytics", "project_management"), "Create a task {s} for {p} and plot engaged users for {s}", ["create_task"]),
]
_PEOPLE = ["Anna", "Carlos", "Mei", "Oliver", "Priya", "Jonas", "Sofia", "Daniel", "Aisha", "Tom"]
_SUBJ = ["budget", "launch", "onboarding", "pricing", "roadmap", "audit", "campaign", "training", "renewal", "hiring"]
_COUNTS = {"analytics": 12, "calendar": 11, "customer_relationship_manager": 8, "email": 9, "project_management": 8}


def mock_tasks(seed: int = 0) -> list[Task]:
    rng = np.random.default_rng(seed)
    tasks, i, tid = [], 0, 0
    specs = []
    for dom, n in _COUNTS.items():
        pool = _SINGLE[dom]
        for k in range(n):
            tpl, w = pool[k % len(pool)]
            specs.append(((dom,), tpl + (" please" if k >= len(pool) else ""), w))
    for doms, tpl, w in _MULTI:
        specs.append((doms, tpl, w))
    for doms, tpl, w in specs:
        for rep in range(10):
            p, s = _PEOPLE[rng.integers(10)], _SUBJ[rng.integers(10)]
            q = tpl.format(p=p, s=s)
            wid = [norm_id(f"{d}.{x}") for x in w for d in doms if any(x == t[0] for t in _MOCK_TOOLS[d])]
            calls = [f"{t.replace('_', '.', 1)}.func()" for t in wid]
            tasks.append(Task(i, q, f"T{tid:02d}", list(doms), calls, wid)); i += 1
        tid += 1
    return tasks
