# Runbook

## 1. 环境准备

```powershell
cd C:\Users\tangerine\.langchain-code-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev]
```

## 2. 准备模型配置

```powershell
Copy-Item .\examples\models.global.example.toml .\models.global.toml
Copy-Item .\examples\auth.example.json .\auth.json
```

编辑：

- [models.global.toml](/C:/Users/tangerine/.langchain-code-agent/models.global.toml:1)
- [auth.json](/C:/Users/tangerine/.langchain-code-agent/auth.json:1)

## 3. 验证配置是否生效

```powershell
lc-agent doctor --config .\tmp.agenttest.config.toml
```

重点检查输出：

- `workspace_root`
- `planner_backend`
- `model_profile`
- `model_sources.model_config_path`
- `model_sources.auth_path`
- `model_sources.model_api_key_source`

## 4. 执行任务

### Dry-run

```powershell
lc-agent run --task "检查仓库结构并搜索 pytest" --workspace C:\Users\tangerine\.langchain-code-agent --mode dry-run
```

### Execute

```powershell
lc-agent run --task "在当前工作空间创建 hello.txt，内容写入 Hello from agent." --workspace C:\Users\tangerine\Desktop\Test\agentTest\test --planner langchain --mode execute --json
```

## 5. 回归测试

```powershell
python -m pytest -q
python -m ruff check src tests
python -m mypy src
```

## 6. 常见问题排查

### `doctor` 显示模型配置为空

检查：

- `models.global.toml` 是否存在
- `default_profile` 是否配置
- profile 下是否包含 `model_backend` 和 `model`

### `doctor` 显示 auth 来源缺失

检查：

- `auth.json` 是否存在
- `auth_ref` 是否与 `auth.json.credentials.*` 一致
- key 是否写在 `model_api_key`

### 执行时报 401 或认证失败

通常表示：

- `auth.json` 里的 key 错误
- 本地模型端点要求 Bearer token，但当前 token 不匹配

### planner 失败

优先看：

- `RunResult.events`
- `FinalReport.errors`
- `attempts[*]`

如果是 `planning_failed`，重点检查：

- 模型端点是否可访问
- planner 输出是否为合法 JSON / structured response

### 工具执行失败

优先看：

- `step_results[*].error_context`
- `final_report.tool_calls`
- `final_report.shell_outputs`

## 7. 安全注意事项

- 不要把真实 key 写回 `models.global.toml`
- 不要提交 `auth.json`
- 如果 `auth.json` 曾进入 Git 历史，应立即轮换 key
