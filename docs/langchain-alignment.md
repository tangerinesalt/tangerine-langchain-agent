# LangChain Alignment Notes

本项目当前对齐 LangChain v1 官方实践的重点如下：

- 规划阶段优先使用 `create_agent(..., response_format=Plan)`。
- `Plan` / `PlanStep` 使用 Pydantic 模型，便于结构化输出校验。
- 模型初始化优先走 `init_chat_model`，保持 provider 解耦。
- 工具层补充了 `@tool` 封装，便于后续直接接入官方 agent 工具调用模式。

保留的工程化偏离：

- 本地 shell 与测试执行仍使用确定性 runner，而不是完全交给通用 agent 循环。
- 这样可以保留白名单、超时和工作目录约束，并让工具层更容易单元测试。
- 对于 `local_http` 模型，planner 使用兼容回退路径，而不是强依赖 tool-calling / structured-output 能力。
