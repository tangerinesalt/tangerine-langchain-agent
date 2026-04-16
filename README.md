# tangerine-langchain-agent
本项目是一个本地代码代理骨架，目标是提供一条可测试、可观察、可逐步扩展的执行链路：

- 通过 CLI 接收任务
- 由 planner 生成结构化计划
- 在受限工作空间内调用本地工具执行
- 记录事件、步骤结果、文件变更和最终报告

当前定位不是“全自治通用代理”，而是“本地可控的代码任务执行框架”。

## 目录概览

```text
src/langchain_code_agent/
  agent/              planner、runner、reporter、validator
  llm/                模型适配与 planner prompt
  models/             Plan / Task / Result 等结构模型
  tools/              文件、shell、测试等本地工具
  workspace/          工作空间与仓库访问边界
  actions.py          action 注册中心
  agent_config.py     代理运行配置入口
  model_resolution.py 全局模型配置与 auth 解析
  cli.py              CLI 入口
```

## 安装

```powershell
cd C:\Users\tangerine\.langchain-code-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev]
```

安装完成后可以使用：

```powershell
lc-agent --help
```

如果没有安装脚本入口，也可以直接使用：

```powershell
python -m langchain_code_agent --help
```

## 配置

配置分成两层：

1. 全局模型配置
2. 工作空间配置

### 1. 全局模型配置

复制示例文件：

```powershell
Copy-Item .\examples\models.global.example.toml .\models.global.toml
Copy-Item .\examples\auth.example.json .\auth.json
```

编辑 [models.global.toml](/C:/Users/tangerine/.langchain-code-agent/models.global.toml:1)：

```toml
default_profile = "local_qwen"

[profiles.local_qwen]
model_backend = "langchain"
model_provider = "openai"
model = "qwen/qwen3.5-9b"
model_base_url = "http://localhost:1234/v1"
model_timeout_seconds = 60
auth_ref = "lmstudio_local"
```

编辑 [auth.json](/C:/Users/tangerine/.langchain-code-agent/auth.json:1)：

```json
{
  "credentials": {
    "lmstudio_local": {
      "model_api_key": "your-real-key"
    }
  }
}
```

注意：

- `auth.json` 已被 `.gitignore` 忽略
- 真实 key 只应放在 `auth.json`
- `models.global.toml` 负责模型 profile，不应存真实 key

### 2. 工作空间配置

工作空间配置只放代理运行参数，不放模型参数。

示例见：

- [examples/config.local.toml](/C:/Users/tangerine/.langchain-code-agent/examples/config.local.toml:1)
- [examples/config.local-http.toml](/C:/Users/tangerine/.langchain-code-agent/examples/config.local-http.toml:1)
- [tmp.agenttest.config.toml](/C:/Users/tangerine/.langchain-code-agent/tmp.agenttest.config.toml:1)

典型内容：

```toml
workspace_root = "C:/Users/tangerine/Desktop/Test/agentTest"
planner_backend = "langchain"

shell_timeout_seconds = 60
max_replans = 1
test_command = "python -m pytest -q"
allowed_shell_commands = ["python", "pytest", "rg", "git"]
```

## 运行

### 检查配置

```powershell
lc-agent doctor --config .\tmp.agenttest.config.toml
```

或：

```powershell
python -m langchain_code_agent doctor --config .\tmp.agenttest.config.toml
```

### 运行一个 dry-run 任务

```powershell
lc-agent run --task "检查仓库结构并搜索 pytest" --workspace C:\Users\tangerine\.langchain-code-agent --mode dry-run
```

### 在指定工作空间执行任务

```powershell
lc-agent run --task "在当前工作空间创建 hello.txt，内容写入 Hello from agent." --workspace ExamplePath --planner langchain --mode execute --json
```

### 使用工作空间配置执行

```powershell
lc-agent run --task "检查 failing tests" --config .\tmp.agenttest.config.toml --mode execute --json
```

## 测试

### 运行全部测试

```powershell
python -m pytest -q
```

### 运行静态检查

```powershell
python -m ruff check src tests
python -m mypy src
```

## 文档

- 架构说明：[docs/architecture.md](/C:/Users/tangerine/.langchain-code-agent/docs/architecture.md:1)
- 运维手册：[docs/runbook.md](/C:/Users/tangerine/.langchain-code-agent/docs/runbook.md:1)
- 已知限制与后续建议：[docs/limitations.md](/C:/Users/tangerine/.langchain-code-agent/docs/limitations.md:1)
- LangChain 对齐说明：[docs/langchain-alignment.md](/C:/Users/tangerine/.langchain-code-agent/docs/langchain-alignment.md:1)
l
python -m pytest
python -m ruff check .
python -m mypy src
```
# tangerine-langchain-agent
l
python -m pytest
python -m ruff check .
python -m mypy src
```
# tangerine-langchain-agent
