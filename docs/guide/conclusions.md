# 总结论

## 一句话定位

**Mooncake** 是把 KV Cache、张量传输、分布式存储和 PD 分离当作第一性问题的 C++ 基础设施。它首先是一个「KV cache-centric distributed system」，其次才是 vLLM/SGLang/LMCache 的后端。

**LMCache** 是把 KV Cache 抽象成可复用知识资产的 Python 生态层。它首先是一个引擎侧缓存管理与插件框架，通过 connector、GPU connector、L1/L2/L3 adapter、多进程 daemon、控制面 coordinator 把 vLLM、SGLang、TRT-LLM 等异构引擎接到多种存储与传输后端。

**vLLM** 是生态接入点。它的 KV Connector V1 把控制面查找/生命周期放到 scheduler，把数据面搬运放到 worker，并用 `MultiConnector` 组合「PD 直接传输」和「共享缓存池」两种路径。

## 最重要判断

1. **数据面厚度差异**：Mooncake 的 Transfer Engine 是完整数据面——Segment/Batch 抽象、内存注册、拓扑矩阵、多 NIC slice spraying、端点池、跨 transport failover。LMCache 更多是把复杂传输委托给 NIXL、UCX、GDS、Mooncake 等后端，自己提供对象生命周期、锁、预取、PD staging 与多进程协调。

2. **存储系统成熟度差异**：Mooncake Store 更像一个「面向张量/对象的分布式缓存数据库」——集中 Master 分配空间、对象多副本 best-effort、租约保护读、软/硬 pin、zombie write 清理、快照与 HA OpLog。LMCache 的存储管理更像插件化缓存框架，本地/L2/P2P 语义强，但跨节点全局强一致语义弱于 Mooncake。

3. **通路选择不是简单三段式**：「H2D 走 PCIe、节点内 D2D 走 NVLink/HCCS、跨机 D2D 走 RDMA」只是 NVIDIA 或 Ascend 常见物理路径，不是两者的统一协议模型。LMCache 通常把通路决策交给 NIXL/UCX/GDS/引擎平台；Mooncake 显式建模拓扑和 transport，可在 RDMA、NVLink、HCCS/HIXL、HIP、TCP、NVMe-oF、GDS 之间选择。

4. **PD 分离路线不同**：LMCache 有两条成熟路线——进程内 vLLM connector + NIXL，以及 MP 模式下独立 daemon + CUDA IPC/SHM/engine-driven 传输；其 chunked PD 设计明确解决了 receiver staging buffer 的部分分配死锁。Mooncake 同时提供 direct Transfer Engine PD connector 和 Mooncake Store 共享池 connector，vLLM 可用 `MultiConnector` 同时启用「一对一低延迟搬运」和「跨实例共享池」。

5. **谁更好取决于目标**：超大规模、跨实例全局缓存池、强生产数据面、异构网络与 RDMA 汇聚带宽——Mooncake 底座更完整。多引擎、多云/多存储、快速生态接入、引擎崩溃不带走缓存、灵活插件化和云原生可观测——LMCache 抽象更好。在「把 KV Cache 当数据库」视角下，Mooncake 的控制面与租约/恢复体系更接近数据库；LMCache 的多进程架构在故障隔离上更现代，但全局一致性尚不如 Mooncake。

## 一句话总结

**Mooncake 负责把 KV Cache 变成集群级状态基础设施；LMCache 负责把 KV Cache 变成引擎外可管理的多租户资产；vLLM 是两者的事实接入协议。**

## 数据库视角评分（摘要）

| 维度 | 更强的一方 | 说明 |
|---|---|---|
| 单对象可见性 | Mooncake | PutStart/PutEnd 保证 complete 后可见 |
| 多副本 | Mooncake | replica best-effort + slice 不同 segment |
| 控制面恢复 | Mooncake | snapshot + OpLog + election + standby catch-up |
| 计算进程故障隔离 | LMCache | MP daemon/reaper/IPC recovery |
| 多租户治理 | LMCache 更细 | cache salt/quota/operator 面向 SaaS 更自然 |
| 持久化分层 | Mooncake | DRAM/SSD/metadata snapshot/OpLog 完整 |

完整评分表见「可靠性与数据库视角」一篇。
