# tangerine-langchain-agent

一个本地运行、可配置、可测试、可调试的代码工程 Agent 骨架。

当前版本提供：

- CLI 入口
- `noop` 与 `langchain` 两种 planner
- `dry-run` 与 `execute` 两种执行模式
- 本地代码仓库访问与文件检索
- 受白名单、超时、工作目录限制的 shell / 测试执行
- 结构化事件日志与结果输出
- 通用模型配置
- 本地 HTTP 模型接入

## 安装

```powershell
cd C:\Users\tangerine\.langchain-code-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev]
```

## 基本运行

不依赖远程模型的最小运行方式：

```powershell
lc-agent run --task "检查仓库并搜索 pytest" --workspace C:\Users\tangerine\.langchain-code-agent --mode dry-run
```

使用默认示例配置执行：

```powershell
lc-agent run --task "检查 failing tests" --config .\examples\config.local.toml --mode execute
```

## 模型配置

### 1. 默认本地模型模式

当前默认配置已经切到本地 HTTP 模型：

```toml
model_backend = "langchain"
model_provider = "openai"
model = "qwen/qwen3.5-9b"
model_base_url = "http://localhost:1234/v1"
model_api_key = "1997"
model_timeout_seconds = 60
```

### 2. LangChain provider 模式

通过 `init_chat_model` 走 LangChain 官方统一模型入口：

```toml
model_backend = "langchain"
model_provider = "openai"
model = "gpt-4o-mini"
model_api_key = "your_api_key"
```

也支持直接写 provider 前缀模型名：

```toml
model_backend = "langchain"
model = "openai:gpt-4o-mini"
model_api_key = "your_api_key"
```

### 3. 本地模型模式说明

对应示例配置见 [config.local-http.toml](/C:/Users/tangerine/.langchain-code-agent/examples/config.local-http.toml:1)。

说明：

- `model_api_key` 是通用字段，不再绑定 `openai_api_key`
- `model_provider` 只在 `langchain` backend 下需要
- `local_http` backend 会直接向配置 URL 发起 POST 请求
- 当前本地模型 planner 走兼容回退路径，要求接口返回可解析文本

## CLI 示例

```powershell
lc-agent doctor --config .\examples\config.local.toml
lc-agent run --task "搜索 repository 类的实现" --workspace C:\Users\tangerine\.langchain-code-agent --mode execute --json
lc-agent run --task "为当前仓库生成执行计划" --config .\examples\config.local-http.toml --planner langchain --mode dry-run --json
```

## 检查

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```
# tangerine-langchain-agent
