# Known Limitations And Next Steps

## 当前限制

### 1. planner 仍然是最不稳定的一环

执行器、结果模型和 completion validation 已经比较稳定，但复杂任务的成功率仍然主要取决于 planner 输出质量
- 提高模型能力、系统约束是一种解决方式。

### 2. action registry 责任偏重

[actions.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/actions.py:1) 当前同时承载：

- action 注册
- 参数校验
- planner schema
- LangChain tool 元数据
- 执行包装

这还可维护，但继续增长会变重。

### 3. 可观测性够用但不深

当前有事件、错误上下文、文件变更和尝试历史，但还缺少：

- traceback 级异常信息
- run_id
- 更细的 planner 分层诊断
- duration 汇总

### 4. 完成判定仍偏启发式

显式 `completion_checks` 已经存在，但很多情况下仍依赖从 plan step 派生检查。

### 5. 目录语义仍有继续优化空间

`agent/` 下 planning 与 execution 仍处于同一级目录，长期演进下可读性会慢慢下降。

## 建议的后续迭代方向

### 低风险高收益

1. 增强可观测性
2. 补更多真实任务验收样例
3. 继续压 planner 错误恢复链路

### 中期结构优化

1. 拆分 `actions.py`
2. 将 `agent/` 明确拆成 planning 和 execution 语义分区

### 不建议立即做的事

1. 不建议现在引入多 agent
2. 不建议马上做大规模目录搬家
3. 不建议为了“更智能”而削弱本地执行边界
