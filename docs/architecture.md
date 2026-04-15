# Architecture

最小骨架采用分层设计：

- CLI: 接收任务与输出结果
- Agent: 生成计划、记录事件日志、按 dry-run/execute 调度工具
- Workspace: 管理仓库根目录与文件访问
- Tools: 统一封装文件检索、shell 和测试命令
- LLM: 可选的 LangChain 规划能力
