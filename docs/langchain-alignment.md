# LangChain Alignment

项目当前与 LangChain 的关系是“局部对齐，执行层保持工程化控制”。

## 已对齐的部分

- 通过 `create_agent(..., response_format=Plan)` 获取结构化 planner 输出
- 使用 `init_chat_model` 适配通用 provider
- 保留 LangChain tool 元数据，便于后续接入更标准的 tool-calling 路径

## 保留的工程化偏离

- shell、测试、文件系统仍由本地确定性执行器控制
- planner 输出不会直接绕过本地校验和工作空间边界
- `local_http` 模式下 planner 仍保留 JSON fallback 路径

## 当前判断

这条路线是合理的，因为项目目标不是“完全让 LangChain agent 直接统治一切”，而是：

- 用 LangChain 提供模型接入与结构化输出能力
- 用本地 runner 保留可测、可控、可约束的执行链

这也是为什么项目里同时存在：

- LangChain planner 路径
- 本地 action registry
- 本地工具执行与 completion validation
