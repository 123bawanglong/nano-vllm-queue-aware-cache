# nano-vLLM Queue-Aware Cache

基于 nano-vLLM 的队列感知 KV Cache 淘汰优化：利用等待队列中的近期前缀需求，减少显存紧张时的缓存淘汰与重复 prefill。

[实验报告与原理详解](docs/experiment.md) · [改动范围与验证依据](docs/changes.md) · [上游 PR #278](https://github.com/GeeeekExplorer/nano-vllm/pull/278)

## 实验结果

RTX 5080 / Qwen3-0.6B / BF16 / 单卡 / CUDA Graph。相同请求轨迹、两组分别预热、交替顺序测量 5 对，取每组算术平均。V0 为同一实现关闭优化，V1 为 `lookahead=16`。

| 指标：重复前缀、68 个 KV blocks | V0 | V1 | 变化 |
|---|---:|---:|---:|
| 输出吞吐（tokens/s） | 394.05 | 442.40 | **+12.27%** |
| 每轮 prefill tokens | 140032 | 107776 | **−23.03%** |
| 平均每轮 p95 TTFT（ms） | 571.16 | 556.03 | −2.65% |
| 每轮调度累计耗时（ms） | 15.48 | 33.97 | +18.50 |

缓存扩大到 160 blocks 后，吞吐变化 −0.20%；无共享前缀时变化 +0.01%，均按测量噪声处理。收益来自缓存压力下减少 prefill，不能外推为通用吞吐提升。TTFT 为离线引擎内从请求波次到达到首 token 的时间，不包含 HTTP 服务链路。

原始数据见 [docs/raw](docs/raw)，报告中的图片为 VS Code 展示已保存日志的截图。上述性能数据来自既有实验，未因仓库整理重新测量。

## 问题与实现

请求结束后，引用归零的物理块进入空闲队列，但其中的 KV 仍可被后续请求复用。按空闲队列顺序分配新块，可能覆盖等待队列中即将使用的缓存前缀。

本实现提供 `prefix_cache_lookahead` 参数，默认 `0` 保持原策略。启用后，在分配即将覆盖有效缓存时：

1. 有界检查等待队列，对候选请求执行只读的连续前缀查询。
2. 在当前分配调用内收集临时保留集合，优先选择集合外的空闲块。
3. 没有其他候选块时回退到原队头，不固定占用块，也不改变请求调度顺序。

`lookahead=16` 限制前 16 个等待条目，包含仍在队头的当前请求；查询跳过当前请求及已有 block table 的请求。保留集合只影响淘汰优先级，引用计数与缓存失效仍走原有分配路径。

## 改动入口

引擎改动集中于 **BlockManager、Scheduler 和配置**三个文件：只读前缀查询、惰性保留集合，以及 Prefill/Decode 分配时的前瞻传递。配套提供 14 项行为测试、成对性能实验和 logits 检查脚本。

阅读 [改动说明](docs/changes.md) 可核对每个文件的职责、设计取舍和验证依据；[核心补丁](docs/queue-aware-eviction.patch) 保留相对上游基线的原始差异，方便区分已有引擎能力与本项目贡献。

## 安装与使用

需要 Linux/WSL、CUDA GPU 和兼容的 PyTorch、Triton、FlashAttention 环境；Python 3.10–3.12。建议使用独立虚拟环境，避免与其他 nano-vLLM 安装冲突。

```bash
git clone https://github.com/123bawanglong/nano-vllm-queue-aware-cache.git
cd nano-vllm-queue-aware-cache
python -m pip install -e .
python -m pip install pytest
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", prefix_cache_lookahead=16)
outputs = llm.generate(
    ["Explain automatic prefix caching."],
    SamplingParams(temperature=0.6, max_tokens=128),
)
print(outputs[0]["text"])
```

## 验证与复现

```bash
python -m pytest tests/test_queue_eviction.py -q
python scripts/check_queue_eviction.py \
  --model /path/to/Qwen3-0.6B --output /tmp/queue-check.json
for scenario in pressure ample unique; do
  python scripts/bench_queue_eviction.py \
    --model /path/to/Qwen3-0.6B \
    --scenario "$scenario" --output "/tmp/queue-$scenario.json"
done
```

14 项行为测试覆盖队列顺序、惰性查询、连续前缀、全部保留时的回退、Decode 分配及随机引用计数检查。GPU logits 检查的对照条件与数值差异见[验证说明](docs/queue_eviction.md)，行为测试通过与性能收益分别验证。

## 目录

```text
src/nanovllm/    推理引擎及缓存淘汰实现
tests/          缓存管理行为测试
scripts/        性能实验与数值检查
docs/           实验报告、原理、截图与原始数据
```

主要实现：[BlockManager](src/nanovllm/engine/block_manager.py)、[Scheduler](src/nanovllm/engine/scheduler.py)、[配置](src/nanovllm/config.py)。

## 来源与许可证

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，保留上游作者信息与 [MIT 许可证](LICENSE)。上游基线为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，优化实现快照为 `e02aa9d6851f4acca47fe091b4dd715674a88334`。队列感知淘汰改动已提交至 [PR #278](https://github.com/GeeeekExplorer/nano-vllm/pull/278)，该链接不表示已合并。

本仓库将 Python 包整理到 `src/`，并相应调整安装配置、测试路径与实验脚本入口；缓存算法沿用上述实现快照。
