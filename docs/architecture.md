# Architecture

## 目标

项目提供一个本地可控的代码代理骨架，重点是：

- 结构化计划
- 受限工具执行
- 可观测的事件与结果
- 可通过测试和真实工作空间做回归验证

## 核心模块

### CLI

- [cli.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/cli.py:1)

职责：

- 接收 `run` 和 `doctor` 命令
- 解析运行参数
- 构建 `AgentConfig`
- 输出 JSON 或可读文本结果

### 配置

- [agent_config.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent_config.py:1)
- [model_resolution.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/model_resolution.py:1)

职责：

- 解析工作空间配置
- 解析全局模型配置 `models.global.toml`
- 解析认证文件 `auth.json`
- 汇总最终运行配置

### Planner

- [agent/planner.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/planner.py:1)
- [agent/plan_output_normalizer.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/plan_output_normalizer.py:1)
- [agent/plan_normalization_rules.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/plan_normalization_rules.py:1)
- [agent/plan_validator.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/plan_validator.py:1)

职责：

- 根据任务生成 `Plan`
- 修复模型输出中的常见格式问题
- 归一化 action 和参数
- 在执行前拒绝明显无效计划

### Execution

- [agent/runner.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/runner.py:1)
- [agent/step_executor.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/step_executor.py:1)
- [agent/completion_validator.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/completion_validator.py:1)
- [agent/run_reporter.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/run_reporter.py:1)
- [agent/replan_context.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/agent/replan_context.py:1)

职责：

- 顺序执行计划步骤
- 记录事件、文件变更和 shell 输出
- 在失败时构建受控 replan 上下文
- 生成最终 `RunResult` 和 `FinalReport`

### Actions

- [actions.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/actions.py:1)

职责：

- 统一注册 action
- 提供参数校验
- 提供 planner 可见的 action schema
- 提供 LangChain tool 元数据

### Tools

- [tools/](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/tools)

职责：

- 文件读写
- 文本搜索
- shell / Python / test 执行
- 时间工具
- LangChain tool 适配

### Workspace

- [workspace/repository.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/workspace/repository.py:1)

职责：

- 工作空间边界控制
- 文件快照
- 文本文件读写
- 忽略规则处理

### 结果模型

- [models/plan.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/models/plan.py:1)
- [models/task.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/models/task.py:1)
- [models/result.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/models/result.py:1)
- [models/replan.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/models/replan.py:1)

职责：

- 统一 planner、runner、reporter 之间的数据结构

## 执行链路

1. CLI 构建 `AgentConfig`
2. `AgentRunner` 创建 `Task`
3. planner 生成 `Plan`
4. normalizer 和 validator 修正/检查 `Plan`
5. `StepExecutor` 逐步执行 action
6. `RunReporter` 记录事件
7. `completion_validator` 校验任务是否真正完成
8. 失败时进入一次受控 replan
9. 输出 `RunResult`

## 当前目录结构评价

当前目录结构整体是清晰的：

- 顶层已经按 `agent / llm / models / tools / workspace` 分区
- 关键链路文件可直接定位
- 测试目录按 `unit / integration` 分开

当前最需要注意的点不是“目录混乱”，而是：

- [actions.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/actions.py:1) 已经承担较多职责
- `agent/` 下 planning 和 execution 文件仍然放在同一级

这两个点目前仍可维护，但属于后续结构演进的主要关注对象。
