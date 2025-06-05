# AlphaDog项目对话记录

## 2024-07-16

### 对话1：策略模块改进

**用户**：`/c:/Users/jimjiang/PycharmProjects/AlphaDog/src/strategy.py#L58-65` 这一段需要可插拔可替换

**助手**：已完成置信区间计算部分的可插拔可替换设计。实现了以下功能：

1. 创建了置信区间计算策略的抽象基类 `ConfidenceIntervalStrategy`
2. 实现了两种具体策略：
   - `NormalDistributionStrategy`：基于正态分布的置信区间计算
   - `QuantileStrategy`：基于分位数的置信区间计算
3. 创建了策略工厂 `StrategyFactory`，支持动态创建和注册策略
4. 修改了 `HighConfidenceReturnStrategy` 类，使其支持从配置文件加载策略
5. 创建了 `strategy_config.yaml` 配置文件，用于配置置信区间计算策略
6. 更新了 `schema.md`，添加了关于可插拔策略设计的说明

现在用户可以通过修改 `strategy_config.yaml` 配置文件来切换不同的置信区间计算策略，而不需要修改代码。