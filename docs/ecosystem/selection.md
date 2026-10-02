---
prev: /ecosystem/recommendations
next: /appendix/benchmark
---

# 选型建议

> 来源：主报告第 12 章。

## 12. 选型建议

### 12.1 优先考虑 Mooncake 的场景

1. 多实例 vLLM/SGLang，负载均衡导致跨节点 prefix 复用；  
2. agent/multi-turn，共享 system prompt、工具上下文和历史会话；  
3. 超大规模 PD 分离，且需要 RDMA/RoCE 多 NIC 汇聚；  
4. 有集群 DRAM/SSD 资源，希望构建独立 cache pool；  
5. 需要 lease/pin/quota/replica/snapshot 的对象生命周期；  
6. MoE、RL、模型权重、hidden state 等张量搬运也想复用同一数据面。

推荐起点：

```text
MooncakeStoreConnector + MooncakeConnector + vLLM MultiConnector
```

但要准备：

- Mooncake Master/HA/metadata 运维；
- RDMA/NIC/NUMA 调优；
- cache pool 容量与驱逐策略；
- connector/vLLM 版本兼容测试。

### 12.2 优先考虑 LMCache 的场景

1. 已经有 vLLM/SGLang/TRT-LLM 多引擎组合；  
2. 希望引擎崩溃/升级不丢 CPU/L2 缓存；  
3. 需要接入 Redis/Valkey、S3、FS、GDS、InfiniStore、Mooncake 等已有存储；  
4. 需要多租户、quota、pin/prefetch/delete、request-level observability；  
5. 平台团队更接受 Python/FastAPI/Kubernetes 生态；  
6. 对 CacheGen/CacheBlend 等非前缀复用有研究兴趣。

推荐起点：

```text
LMCache MP mode + LocalCPU/L2 + NIXL PD
```

但要准备：

- MP daemon 资源与版本管理；
- CUDA IPC/SHM/GDS 权限与生命周期；
- 后端语义差异测试；
- coordinator durable replay 规划。

### 12.3 两者组合的场景

LMCache 已有 Mooncake Store L2 adapter，`lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py` 支持把 Mooncake 接成 LMCache 的 L2。这个组合有意义：

- LMCache 管 engine layout、多引擎、MP daemon、多租户；
- Mooncake 管跨节点池、RDMA、对象生命周期和 SSD；
- 平台可以先保留 LMCache 抽象，逐步把冷层/远层迁到 Mooncake。

代价是边界更多：LMCache key、Mooncake object key、GPU block pin、L2 lookup lock、quota 与驱逐策略都要测试。

### 12.4 不建议的选择方式

1. 不要只看峰值带宽；  
2. 不要把 CacheGen/CacheBlend 当成生产默认；  
3. 不要把 Mooncake best-effort replica 当强持久化；  
4. 不要把 LMCache coordinator 事件投影当成强一致控制面；  
5. 不要忽略 vLLM 版本差异和 HMA/hybrid 模型限制；  
6. 不要在没有 RDMA/NIC 拓扑验证的机器上直接外推官方 benchmark。

---
