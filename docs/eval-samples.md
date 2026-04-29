# Eval Samples

The default eval suite is intentionally stable. It only loads JSON files directly
under `tests/fixtures/agent_tasks`:

```powershell
python -c "from pathlib import Path; from langchain_code_agent.evals.runner import run_eval_suite; project=Path.cwd(); case_paths=sorted((project/'tests/fixtures/agent_tasks').glob('*.json')); report=run_eval_suite(case_paths, project_root=project, workspaces_root=project/'.lca/evals/workspaces', report_path=project/'.lca/evals/latest.json'); print(report.model_dump_json(indent=2))"
```

Optional samples live in subdirectories and must be selected explicitly.

## Real Planner Samples

These cases use `planner_backend = "langchain"` and do not include preset plans.
They require a working model configuration through `models.global.toml`,
environment variables, or the optional case-level model fields.

```powershell
python -c "from pathlib import Path; from langchain_code_agent.evals.runner import run_eval_suite; project=Path.cwd(); case_paths=sorted((project/'tests/fixtures/agent_tasks/real_planner').glob('*.json')); report=run_eval_suite(case_paths, project_root=project, workspaces_root=project/'.lca/evals/real-planner-workspaces', report_path=project/'.lca/evals/real-planner.json'); print(report.model_dump_json(indent=2))"
```

Current real planner samples:

- `real-planner-create-snake-game`
- `real-planner-update-readme-section`

## Failure Mode Samples

These deterministic cases broaden failure coverage without external model calls.

```powershell
python -c "from pathlib import Path; from langchain_code_agent.evals.runner import run_eval_suite; project=Path.cwd(); case_paths=sorted((project/'tests/fixtures/agent_tasks/failure_modes').glob('*.json')); report=run_eval_suite(case_paths, project_root=project, workspaces_root=project/'.lca/evals/failure-mode-workspaces', report_path=project/'.lca/evals/failure-modes.json'); print(report.model_dump_json(indent=2))"
```

Current failure mode samples:

- `failure-detects-failed-tests`
- `failure-detects-unexpected-file-change`
- `failure-classifies-provider-timeout`
- `failure-normalizes-list-files-root-path`

## Maintenance Rules

- Keep default cases deterministic and model-free.
- Put real planner cases under `real_planner`.
- Put optional failure expansions under `failure_modes`.
- Use `content_contains` for generated files when exact full content would be too
  brittle.
- Keep every case focused on one primary behavior.
