# Known Limitations And Next Steps

## 当前限制

### 1. planner 仍然是最不稳定的一环

执行器、结果模型和 completion validation 已经比较稳定，但复杂任务的成功率仍然主要取决于 planner 输出质量
- 提高模型能力、系统约束是一种解决方式。

### 2. 真实任务覆盖仍偏少

当前已有 unit、integration 和 eval fixture，但真实工作空间任务样本仍有限。复杂任务成功率需要更多真实失败样本来驱动 planner prompt、normalizer 和 repair 规则演进。

### 3. action registry 后续会继续增长

[action_registry/registry.py](/C:/Users/tangerine/.langchain-code-agent/src/langchain_code_agent/action_registry/registry.py:1) 已经从 `actions.py` 拆出，当前可维护；但随着工具数量增加，注册表可能继续变大。后续可以按工具族拆分，同时保持 action schema、参数校验和执行分发的一致性。

### 4. 可观测性够用但还可继续产品化

当前有事件、错误上下文、traceback、run_id、文件变更、尝试历史和 artifact。后续更值得补的是：

- 更方便查询的失败索引
- 更稳定的 run artifact schema 文档
- 更细的 planner 分层诊断视图

### 5. 完成判定仍偏启发式

显式 `completion_checks` 已经存在，但很多情况下仍依赖从 plan step 派生检查。

## 建议的后续迭代方向

### 低风险高收益

1. 补更多真实任务验收样例
2. 继续压 planner 错误恢复链路
3. 增强 artifact 查询和失败诊断

### 中期结构优化

1. 按工具族拆分 `action_registry/registry.py`
2. 为 run artifact 补稳定 schema 文档

### 不建议立即做的事

1. 不建议现在引入多 agent
2. 不建议继续做大规模目录搬家
3. 不建议为了“更智能”而削弱本地执行边界
