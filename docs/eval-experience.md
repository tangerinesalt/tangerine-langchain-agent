# Eval Experience Archive

The experience archive is a deterministic layer on top of the eval harness. It
does not change planner behavior, runner behavior, shell permissions, workspace
boundaries, or default CLI execution.

Its job is to preserve facts from eval runs:

- eval case id, task, expected outcome, and expected actions
- observed actions, errors, failure codes, repair codes, and artifact paths
- planner repair records from `plan_repaired` events
- a lightweight JSON index for local analysis

It intentionally avoids free-form model summaries. Every archived value comes
from an `EvalCase`, an `EvalReport`, or the run artifact referenced by the
report.

## Build an Archive

```powershell
lc-agent eval archive --json
```

The command writes:

- `.lca/evals/experience/records.jsonl`
- `.lca/evals/experience/index.json`
- `.lca/evals/experience-report.json`

## Run the Eval Suite

```powershell
lc-agent eval run --json
```

Use explicit paths when running from outside the project root:

```powershell
lc-agent eval run --project-root C:\Users\tangerine\.langchain-code-agent --cases tests/fixtures/agent_tasks --workspaces .lca/evals/workspaces --report .lca/evals/latest.json --json
```

## Query Records

```python
from langchain_code_agent.evals.experience import (
    load_experience_records,
    query_experience_records,
)

records = load_experience_records(".lca/evals/experience/records.jsonl")
planning_failures = query_experience_records(
    records,
    failure_code="missing_workspace_path",
)
repairs = query_experience_records(
    records,
    repair_code="append_run_tests_verification",
)
```

## Record Types

`success` means the eval expectation passed and the agent reported success.

`expected_failure` means the eval expectation passed because the agent correctly
failed, such as a blocked shell command or a missing workspace path.

`expectation_mismatch` means the eval expectation itself did not pass and should
be investigated before using the record as a baseline.

## Maintenance Notes

- Treat `records.jsonl` as an analysis artifact, not as planner context.
- Keep schema versions explicit when adding fields.
- Add new index keys only when they are derived from existing structured fields.
- Prefer adding eval cases for new repeated failures before relying on archive
  analysis.
