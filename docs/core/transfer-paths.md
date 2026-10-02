---
prev: /core/architecture
next: /core/zero-copy

---

# 物理通路：H2D、节点内 D2D、跨机 D2D

「H2D 走 PCIe、节点内 D2D 走 NVLink/HCCS、跨机 D2D 走 RDMA」这个常见三段式假设对两个系统都成立吗？本篇给出逐通路对比与直接回答。底层地址/注册/零拷贝机制见下一篇。

> 来源：主报告第 2 章全文。


> **🎯 面试考察点**（66 家公司真题库 · 物理通路）：
> - 「PD 分离 KV 传输开销大为什么还值得？」——×3 问法；
> - 追问链：三段式假设哪里不对 → 同机架原则与带宽公式 → RoCE/IB 选型 → 传输量 vs 重算量的规模定律。
> 精答见[面试题库 Q11](/interview/inference-answers)与[CUDA 篇 Q15](/interview/cuda-systems-answers)

## 2. 物理通路：H2D、节点内 D2D、跨机 D2D

用户问题中的典型假设是：

```text
H2D/D2H：PCIe
节点内 D2D：NVIDIA NVLink，或 Ascend HCCS
跨机 D2D：RDMA / RoCE
```

这个模型对 NVIDIA 多机推理大体成立，但对 LMCache 与 Mooncake 的软件抽象都不完整。

### 2.1 LMCache：通路抽象委托，路径策略分散

LMCache 没有自研统一拓扑管理器，而是分成几类路径。

#### 2.1.1 引擎本地 H2D/D2H

`lmcache/v1/gpu_connector/gpu_connectors.py` 中的 vLLM connector 使用 PyTorch tensor copy、临时 GPU staging buffer 和 `non_blocking=True`。`from_gpu()` 常见路径是 paged KV 拷到 GPU staging，再拷到 CPU/L1 `MemoryObj`；`to_gpu()` 反向执行。V2/V3 connector 的差异主要是 vLLM KV layout、指针与 kernel-facing descriptor 组织方式不同。

这部分通常经过 PCIe，但 LMCache 不显式声明“必须 PCIe”。在 GDS 路径中，存储到 GPU 可以走 GPUDirect Storage；在 CUDA IPC 的 MP 模式中，同机 GPU 之间又可能利用 CUDA IPC 和设备侧拷贝。

#### 2.1.2 本地 SSD/NVMe

LMCache 提供：

- Local disk；
- GDS backend；
- NIXL Store/GDS 路径；
- ROCm/HIP、SYCL/XPU、HPU、MUSA 等平台 connector。

`lmcache/v1/storage_backend/gds_backend.py` 显示 LMCache 会加载 CUDA/ROCm runtime copy 函数，通过 GPUDirect Storage 绕过或减少 CPU staging。也就是说，NVMe 到 GPU 不必然经过传统 H2D PCIe staging。

#### 2.1.3 跨节点与 P2P

LMCache P2P 的 `Transfer Channel` 当前明确只有 NIXL 实现，NIXL 后端常用 UCX/RDMA：

- `docs/design/v1/distributed/transfer_channel/overall.md`：peer 注册 L1 buffer，ZMQ 交换 metadata，之后 submit read/poll；
- `docs/design/v1/distributed/l2_adapters/p2p_l2_adapter.md`：lookup-and-lock -> query address -> RDMA read -> unlock；
- `examples/p2p/README.md` 明确生产性能要求 RDMA fabric，并提到 UCX `rc` transport。

LMCache P2P 是只读 pull：从 peer 的 L1 CPU 内存读对象，不做远程写。这降低了写一致性和远程故障破坏本地的风险，但容量仍受集群 DRAM 限制。

#### 2.1.4 LMCache 与 Ascend/HCCS

在当前源码中，LMCache 主仓库没有原生 HCCS transport 或 HCCS 拓扑选择。Ascend 场景更多依赖平台 connector、引擎生态或 Mooncake/NIXL 等后端。因此不能说“LMCache 也原生走 HCCS”。更准确的说法是：

- NVIDIA：H2D/D2H 通常由 connector/引擎拷贝承担，跨节点由 NIXL/UCX/RDMA 承担，节点内可由 UCX/CUDA IPC/GDS 决定；
- Ascend：LMCache 自身没有一等 HCCS 数据面，若接 Mooncake 或引擎侧 Ascend 组件，才可能获得 HCCS 路径。

### 2.2 Mooncake：通路是 Transfer Engine 的核心建模对象

#### 2.2.1 Segment / Buffer / BatchTransfer

`Transfer Engine` 的基本模型：

- Segment 表示本机地址空间或 NVMe-oF 空间；
- Buffer 是注册给 RDMA/GPUDirect 的连续地址范围；
- BatchTransfer 是一组异步 read/write 请求；
- 应用调用 `submitTransfer()` 后用 `getTransferStatus()` 轮询。

文档明确 RAM Segment 可以覆盖 DRAM/VRAM，NVMeoF Segment 可直接连接远端 NVMe。RDMA 支持 DRAM/VRAM 到远端 DRAM；cuFile/GDS 支持 DRAM/VRAM 与 NVMe。

#### 2.2.2 NVIDIA / NVLink / RDMA

`docs/source/design/transfer-engine/index.md` 描述：

- 拓扑矩阵按 `cpu:*`、`cuda:*` 与 HCA 生成 preferred/secondary NIC 列表；
- GPU buffer 尽量选择同一 NUMA 或同一 PCIe switch 下、支持 GPUDirect RDMA 的 NIC；
- 长请求切成多个 slice，slice 可落到不同 RDMA NIC 聚合带宽；
- endpoint 池限制连接数量，失败时移除并重建；
- TENT 增加了基于 EWMA 的 slice spraying、NUMA penalty、transport fallback 和 rail failover。

因此 Mooncake 不是简单“跨机固定 RDMA”，而是显式选择“哪个 GPU/NIC/NUMA/slice/transport”。

#### 2.2.3 Ascend / HCCS

`docs/source/design/transfer-engine/ascend_direct_transport.md` 非常明确：

- Ascend Direct Transport 基于 CANN ADXL；
- 支持 Host-to-Device、Device-to-Host、Device-to-Device；
- 通信协议包括 HCCS 与 RDMA；
- A2 server/A3 supernode 内默认 HCCS；
- 设置 `HCCL_INTRA_ROCE_ENABLE=1` 可改为 RDMA；
- HCCS 要求设备内存按 2MB 页表对齐；
- A3 的 fabric memory 模式可让 Store 访问远端 host memory。

旧 Ascend Transport 文档也说明系统会判断是否跨 HCCS，并选择 HCCS 或 ROCE。新 Ascend Direct Transport 已被推荐替代旧实现。

结论：**Mooncake 在 Ascend 上已经把 HCCS 作为 transport 一等公民；在 NVIDIA 上把 NVLink/GPUDirect RDMA/RDMA multi-rail 作为一等拓扑决策。**

### 2.3 vLLM：提供通路钩子，不负责统一数据面

vLLM 在 KV Connector V1 中有 `set_host_xfer_buffer_ops()`，类型是 `CopyBlocksOp`，方向枚举为 `h2d` 和 `d2h`。这说明 vLLM 允许 connector 在需要 host/device buffer 时注入平台专用拷贝操作。

vLLM 还支持：

- `VLLM_GPU_NIC_PCIE_MAPPING`；
- `VLLM_NIC_SELECTION_VARS`；
- 在 executor 中把 GPU BDF 映射到 NIC BDF。

这说明 vLLM 认识到 RDMA NIC 与 GPU 的 PCIe/NUMA 亲和，但具体路径由 connector 后端决定。vLLM 本身不做 LMCache 或 Mooncake 那种完整 KV cache 拓扑调度。

### 2.4 直接回答：是否都这么做？

| 通路 | LMCache | Mooncake | 结论 |
|---|---|---|---|
| H2D/D2H | 常经 PCIe，但可有 CUDA IPC/GDS/平台专用 connector；通路不内建统一声明 | NVIDIA 常经 PCIe/cudaMemcpy/GPUDirect；Ascend Direct 明确支持 H2D/D2H；GDS/NVMe-oF 可改变传统路径 | 两者不都简单走 PCIe |
| 节点内 D2D | NVIDIA 可经 NIXL/UCX/CUDA IPC；LMCache 主仓库无 HCCS 一等实现 | NVIDIA 支持 NVLink/intra-NVLink transport；Ascend 支持 HCCS；可自动或配置选择 | Mooncake 显式；LMCache 主要委托后端 |
| 跨机 D2D | P2P 当前走 NIXL/UCX RDMA；Mooncake backend 走 Mooncake TE/RDMA | 支持 RDMA/RoCE、multi-NIC slice、failover；也支持 TCP/EFA/NVMe-oF 等 | 两边都把 RDMA 当主生产路径，但 Mooncake 的拓扑控制更深 |

---

## 附：LMCache 传输协议栈全景（一研系列，已源码复核）

> 来源：「一研」《LMCache 传输协议详解：从 PCIe 到 RDMA 的协议栈全景》，本地快照 `references/raw/wechat_2_utf8.txt`。核心数字与代码引用已对照 LMCache 源码验证（证据等级 C+）。

LMCache 的传输协议栈分四层：

| 层次 | 协议 | LMCache 实现 |
|---|---|---|
| 物理链路层 | PCIe / NVLink / IB / RoCE | 硬件提供 |
| 硬件传输层 | CUDA Stream / RDMA Verbs / POSIX | GPUConnector / NixlChannel |
| 传输框架层 | NIXL / UCX | NixlChannel / NixlStoreL2Adapter |
| 应用协议层 | ZMQ / Redis / S3 / msgpack | PDBackend / RemoteBackend |

设计哲学：**用抽象换简洁**（NIXL 统一 NVLink/RDMA/GDS 的 API）、**用异步换吞吐**（CUDA Stream 让 D2H/H2D 不阻塞计算）、**用分层换灵活**（数据面用最快的 RDMA/NVLink，控制面用最可靠的 ZMQ）。

### PCIe + CUDA Stream：节点内通路

PCIe 带宽远低于 HBM（PCIe 4.0 x16 约 31.5 GB/s vs HBM 约 3000 GB/s），D2H/H2D 时间几乎完全由 PCIe 决定：

| 数据量 | PCIe 4.0 | PCIe 5.0 |
|---|---|---|
| 1 MB | ~31 μs | ~16 μs |
| 100 MB | ~3.1 ms | ~1.6 ms |
| 1.5 GB（70B 4K token KV Cache） | ~47 ms | ~24 ms |

LMCache 三个优化：pinned memory（消除 pageable 中转）、批量传输（合并小传输）、独立 `load_stream`（`torch.cuda.Stream()`，不阻塞引擎默认 stream）。LayerwiseGPUConnector 用 Python generator 逐层传输，GPU 计算第 N+1 层时 load_stream 已在搬第 N 层。多硬件 Stream 适配：NVIDIA/ROCm `torch.cuda.Stream`、Intel XPU `torch.xpu.Stream`、HPU `mark_step()`、MUSA `torch.musa.Stream`。

### RDMA：两条使用路径

- **NixlChannel**（NIXL 封装）：`make_prepped_xfer("WRITE", ...)` → `transfer(handle)` → 1ms 轮询 `check_xfer_state`；PD 分离主要用 RDMA Write（Push），P2P 用 Read/Write 双向；
- **InfiniStoreConnector**（直接 RDMA）：`rdma_write_cache_async` / `rdma_read_cache_async`。

前置条件都是 Memory Registration：注册预分配缓冲区（物理地址+大小+密钥），GPU Direct RDMA 则通过 NIXL 注册 VRAM 类型实现。

### NIXL：后端自动选择

LMCache 默认 UCX Backend：同节点 GPU→GPU 走 NVLink、同节点 CPU→CPU 走共享内存/POSIX、跨节点有 IB/RoCE 走 RDMA、无 RDMA 时 TCP 兜底。NIXL 存储后端族：GDS / GDS_MT / POSIX / HF3FS / OBJ / AZURE_BLOB / DOCA_MEMOS（由 `NixlStoreL2Adapter` / `NixlStorageBackend` 使用）。

### ZMQ 控制面

REQ/REP（PD 内存分配/缓存查询/P2P Lookup）、PUSH/PULL（ProxyNotif 传输完成通知）、ROUTER/DEALER（MP RPC）；IPC 用于同节点、TCP 用于跨节点。消息用 msgspec msgpack 编码（`tag=True` 启用多态反序列化——一条消息可能是 AllocRequest/AllocResponse/ProxyNotif，按标签自动选择类型）。REQ/REP 要求严格 send-recv 交替，超时后必须关闭重建（见「源码审计要点」的适用范围说明：5s 超时特指 pd_backend 的 cache-query socket）。

### 存储协议细节

- **Redis**：String 类型，每 key 拆 `kv_bytes` + `metadata` 双条目；`asyncio.Semaphore(150)` 限并发连接；
- **S3**：上传用 MemoryViewStream 包装 memoryview 零拷贝，下载用 ctypes.memmove 直写 MemoryObj；
- **GDS**：`CuFile` 直接以 GPU 指针读写 NVMe，`CuFileMemoryAllocator(align_bytes=4096)` + `cuFileBufRegister`。

### 协议性能对比（1.5 GB KV Cache）

| 协议 | 物理网络 | 传输时间 | CPU 参与 |
|---|---|---|---|
| PCIe + Pinned | PCIe 4.0 | ~47 ms | 有（DMA） |
| PCIe + Pageable | PCIe 4.0 | ~60 ms | 有（拷贝+DMA） |
| NVLink | NVLink 4.0 | ~0.8 ms | 无 |
| RDMA Write/Read | IB NDR | ~30 ms | 无 |
| GPU Direct RDMA | IB NDR | ~15 ms | 无 |
| GDS Write | NVMe | ~200 ms | 无 |
| POSIX Write | NVMe | ~300 ms | 有 |
| Redis SET | TCP 10G | ~1.2 s | 有 |
| S3 PUT | 互联网 | ~5-30 s | 有 |

选择原则一句话：**快路径优先——能用 NVLink 不用 PCIe，能用 RDMA 不用 TCP，能用 GPU Direct 不用 CPU 中转**；UCX 在运行时自动选择最优路径，上层代码无需关心底层差异。

## 附二：物理网络拓扑与 PD 可行性推算（一研「物理网络详解」篇）

协议栈之外，**物理层拓扑**同样决定 KV 传输性能：

- **NUMA 腰斩**：GPU 挂不同 NUMA 节点时 PCIe D2H/H2D 带宽可能腰斩——LMCache 用 `NUMADetector` + `alloc_pinned_numa_ptr` 绑定 GPU 所在 NUMA；
- **机架层级代价表**：同节点 NVLink 900GB/s ~0.1μs → 同机架（1 跳 Leaf）400Gb/s ~1μs → **跨机架（Leaf-Spine-Leaf 3 跳）200Gb/s ~3-5μs（带宽减半）** → 跨集群 10μs+。**核心原则：PD 的 Prefill 与 Decode 应部署在同一 Leaf 交换机下**；
- **PD 带宽规划公式**：所需 RDMA 带宽 = KV 大小 / 目标传输延迟。70B 4K：1.5GB / 30ms = 50GB/s = 400Gb/s——IB NDR 刚好满足，建议预留 20%；
- **GPU Direct RDMA 四项网络前置**：GPU BAR 空间被 RNIC 映射（nvidia-peermem）、IOMMU 不干扰 DMA、PCIe ATS、RNIC 与 GPU 同一 PCIe Switch（`nvidia-smi topo -m` 检查）；
- **PD 收益的规模定律**：KV 传输线性（带宽决定）而 Prefill 计算超线性（token 二次增长）——8B 3ms vs 30ms 重算（✅）、70B 30ms vs 200ms（✅）、405B-32K 1280ms vs 8000ms（✅）——**模型越大上下文越长 PD 收益越明显**；GPU Direct RDMA 把 70B 4K 的 124ms 传统路径压到 15ms；
- IB 运维四特性：PFC（配错引发风暴瘫痪集群）、ECN+DCQCN（RDMA 拥塞）、Subnet Manager（节点变更重路由）、Partition Key（多集群隔离）；RoCE 无损配置（pfc/ecn/mqprio 命令集）与「生产 PD 用 IB、测试用 RoCE」的经验法则；
- **RoCE 演进五阶段**（一研 RoCE 系列）：IB 专用封闭期（3-5× 成本）→ RoCE v1（2007，EtherType 直封以太帧，**无 IP 头不可路由**，仅同 VLAN）→ **RoCE v2（UDP/IP 封装、端口 4791、三层可路由**——从专用协议变 IP 网络公民）→ 无损以太网三件套（**PFC(802.1Qbb) + ECN + DCQCN**：交换机超阈值标 ECN → 接收方发 CNP → 发送方指数降速线性恢复；PFC 双刃剑——防丢包但引入队头阻塞，配错可致风暴）→ AI 爆发（400G 标配）。RDMA 丢包敏感：性能可从 100GB/s 掉到 10GB/s。**Kimi 生产用 RoCE v2（4×200G ConnectX-6/7），选 RoCE 的核心是成本与复用 IP 运维体系而非性能**（IB 延迟 0.6-1μs vs RoCE 1-2μs）。
