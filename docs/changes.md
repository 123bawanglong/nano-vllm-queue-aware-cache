# 改动范围与验证依据

本项目在 nano-vLLM 的已有推理引擎上改进空闲 KV block 的选择策略，贡献范围包括淘汰逻辑、调度器衔接、行为测试和对照实验。

## 相对上游的核心改动

基线：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`。优化快照：`e02aa9d6851f4acca47fe091b4dd715674a88334`。

| 文件 | 原行为 | 本项目改动 |
|---|---|---|
| [config.py](../src/nanovllm/config.py) | 无等待队列前瞻参数 | 增加非负 `prefix_cache_lookahead`，默认 0 关闭 |
| [block_manager.py](../src/nanovllm/engine/block_manager.py) | 从 free 队头获取新块 | 只读查询候选请求的连续缓存前缀；惰性构建保留集合；优先取集合外的块，必要时回退 |
| [scheduler.py](../src/nanovllm/engine/scheduler.py) | 分配器不接收等待队列信息 | 用 `islice` 提供有界前瞻，在 Prefill 分配和 Decode 扩块时传入 |
| [test_queue_eviction.py](../tests/test_queue_eviction.py) | 无该策略的专项测试 | 添加 14 项行为测试，覆盖回退、只读性、连续前缀和引用计数等 |
| [bench_queue_eviction.py](../scripts/bench_queue_eviction.py) | 无该策略的成对对照实验 | 固定输入轨迹、物理 KV 容量、预热和交替运行顺序，测量吞吐、TTFT、prefill 与调度时间 |
| [check_queue_eviction.py](../scripts/check_queue_eviction.py) | 无该策略的专项数值诊断 | 比较完整词表 logits，并分别报告相同缓存命中条件和缓存压力基线下的差异 |

三个引擎文件合计增加 43 行、删除 8 行；不包含测试、脚本和文档。具体改动可查看 [上游 PR #278 的文件差异](https://github.com/GeeeekExplorer/nano-vllm/pull/278/files)。

## 关键设计选择

- **只在需要时查队列。** 当原队头没有有效缓存时，不构建保留集合；同一次分配调用内最多构建一次。
- **沿用连续前缀约束。** 链式 hash 与 token 内容校验遇到第一次 miss 就停止，不跳过缺失块复用后面的块。
- **软保留。** 保留集合不增加引用计数，不预留容量；所有候选块都被保留时仍正常分配。
- **不重排请求。** 改动集中于空闲块选择，不改变 admission 顺序。已有的引用计数和旧 hash 映射失效路径仍负责维护状态。
- **默认关闭。** 使用者显式启用，缓存无压力或前缀不共享时无需承担额外查询成本。

代价是等待前缀的 CPU hash/查询，以及 deque 上的扫描和移除。前瞻限制的是请求条目数，不是总前缀 token 数；较大 free 队列下连续分配最坏可能出现平方级扫描成本。本实现尚未证明大缓存或生产服务场景的扩展性。

## 证据如何对应结论

| 要验证的结论 | 证据 | 能说明的范围 |
|---|---|---|
| 缓存状态更新符合预期 | 14 项行为测试，包含 1000 次随机操作 | 已覆盖场景中的逻辑与账目一致性 |
| 压力场景减少 prefill 并提高吞吐 | [pressure 原始数据](raw/pr-pressure.json) | 单模型、单卡、固定种子的 5 对离线实验 |
| 收益与缓存竞争有关 | [ample](raw/pr-ample.json)、[unique](raw/pr-unique.json) | 两类控制场景中的差异低于 1%，按噪声处理 |
| 特定条件下数值一致 | [logits 检查](raw/pr-correctness.json) | 相同缓存命中对照下 64 行 logits 逐位一致；不声称与压力基线逐位一致 |
| Decode 跨块后引用正常释放 | [长输出记录](raw/pr-long-output.json) | 已测的 16 个请求、每请求 256 输出 tokens |

测试通过不等于吞吐提升；性能结果也不替代正确性检查。两者使用独立脚本验证。

## 仓库整理

发布仓库将 `nanovllm/` 移至 `src/nanovllm/`，调整 setuptools 的包发现、pytest 路径以及 benchmark 导入入口。引擎源码与上述优化快照保持一致；目录整理不计入算法改动。

上游模型实现、Attention 算子和基础推理能力来自 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)。本仓库保留其作者信息和 [MIT 许可证](../LICENSE)。
