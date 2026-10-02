# 术语表

| 术语 | 全称/含义 | 说明 |
|---|---|---|
| KV Cache | Key-Value Cache | Transformer 推理中缓存的注意力 K/V 张量，用于避免重复 Prefill 计算 |
| Prefill / Decode | — | 推理两阶段：Prefill 处理完整输入（compute-bound），Decode 逐 token 生成（memory-bound） |
| PD 分离 | Prefill-Decode Disaggregation | 将 Prefill 与 Decode 部署到不同节点，KV Cache 需跨节点传输 |
| PagedAttention | — | vLLM 借鉴 OS 分页思想管理 KV Cache 的机制，块为最小分配单元 |
| APC | Automatic Prefix Caching | vLLM 自动前缀缓存，按 block hash 链复用相同前缀的 KV |
| RadixAttention | — | SGLang 用基数树组织 prefix 的缓存索引结构 |
| MHA/GQA/MQA/MLA | 多头/分组查询/多查询/多头潜在注意力 | 注意力变体，决定 KV Cache 的头数与体积 |
| H2D/D2H | Host-to-Device / Device-to-Host | CPU 与 GPU 之间的数据搬运方向 |
| D2D | Device-to-Device | GPU 之间（节点内 NVLink/HCCS 或跨机 RDMA）的数据搬运 |
| GDS | GPUDirect Storage | GPU 与 NVMe 之间绕过 CPU staging 的直通路径 |
| GPUDirect RDMA | — | NIC 与 GPU 显存之间绕过 CPU 的 RDMA 直通 |
| HCCS | Huawei Cache Coherence System | 华为昇腾节点内互连，对标 NVLink |
| MR | Memory Region | 注册到 NIC 可被 RDMA 访问的内存区域（含 lkey/rkey） |
| QP/CQ | Queue Pair / Completion Queue | RDMA 的请求队列与完成队列 |
| NIXL | NVIDIA Inference Exchange Layer | NVIDIA 的传输/存储抽象层，LMCache P2P/PD 使用 |
| UCX | Unified Communication X | 通信框架，NIXL 常用后端，自动选择 NVLink/RDMA/TCP |
| TE | Transfer Engine | Mooncake 自研数据面：Segment/Buffer/Batch/多 NIC/拓扑感知 |
| TENT | Transfer Engine New Topology(传输增强) | Mooncake TE 的 transport selector/QoS/slice spraying/failover 演进 |
| Segment/Buffer | — | Mooncake TE 的地址空间与注册内存抽象 |
| Slice | — | Mooncake Store 对象的切分单元（约 16MB 上限），可分布到不同节点 |
| Replica | — | Mooncake 对象的多副本（best-effort，非强一致） |
| PutStart/PutEnd | — | Mooncake 对象写协议：分配空间→写数据→发布可见 |
| Lease | — | Mooncake 读保护：lease 活跃期间对象不可被驱逐/删除 |
| Soft/Hard Pin | — | Mooncake 驱逐优先级：软 pin 低优先级驱逐，硬 pin 永不驱逐 |
| OpLog | — | Mooncake Master HA 的操作日志，standby 追赶用 |
| MemoryObj | — | LMCache 的带形状物理值对象（tensor+shape+dtype+format） |
| CacheEngineKey | — | LMCache 的 chunk 身份键：model@world_size@worker_id@chunk_hash@dtype |
| L1/L2/L3 | — | LMCache 缓存层级：本地 CPU / 分布式后端 / 远端存储（不同语境下分层定义有差异） |
| MP 模式 | Multiprocess | LMCache 独立缓存 daemon 模式，引擎崩溃不带走缓存 |
| Coordinator | — | LMCache fleet 级控制面：registry/事件流/quota/pin/prefetch |
| Reservation | — | LMCache PD 的 admission control：receiver 预留整个请求的 staging 空间 |
| CUDA IPC | — | NVIDIA 跨进程 GPU 内存访问机制，LMCache MP 低延迟路径 |
| Connector | — | vLLM KV Connector V1：外部缓存系统接入 scheduler/worker 的协议边界 |
| MultiConnector | — | vLLM 组合多个 connector 的机制，如 direct PD + Store 共享池 |
| HMA | Hybrid Memory Allocator / hybrid manager | vLLM 混合注意力/混合架构下的内存分组管理 |
| KVBM | KV Block Manager | NVIDIA Dynamo 内的 KV 块管理组件 |
| HiCache | — | SGLang 的分层缓存（L1 GPU/L2 CPU/L3 远端） |
| CacheGen/CacheBlend | — | LMCache 的预研方向：KV 压缩流式传输 / 非前缀复用与部分重算 |
| chunk | — | LMCache 的可缓存逻辑单元：若干 token 的 KV 组成，键为 chunk hash |
| block | — | vLLM paged KV 的物理单元，block_size 常为 16 |
| store_job_id | — | vLLM Mooncake Store 的异步写池任务标识，持有 GPU block 引用 |
| cache salt | — | 租户/部署隔离的键命名空间前缀，防跨租户 key 冲突 |
