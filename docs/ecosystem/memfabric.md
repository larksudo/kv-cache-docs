---
prev: /ecosystem/sglang-ascend-pooling
next: /ecosystem/source-audit
---

# MemFabric Hybrid 深度分析：昇腾的内存池化底座

> 前几章我们多次提到 MemFabric（vllm-ascend 的 MemcacheBackend、sparse decode offload、GVA、`MF_DEVICE_UB_QOS`）——本篇把它作为分析对象：华为昇腾开源的**内存池化软件**（gitcode.com/Ascend/memfabric_hybrid，2025/11 开源），DRAM+HBM 混合池化、全局统一编址、跨机内存直接访问。全部机制对照源码（`src/hybm`/`src/smem`/`src/acc_links`/`src/acc_offload`）。

## 1. 定位：它不是 KV Cache 管理器，是内存语义层

先纠正一个容易混淆的点：MemFabric **没有**对象、缓存、驱逐、租约、前缀任何 KV 语义——它回答的唯一问题是「**怎么让分布在不同节点不同介质（HBM/DRAM）的内存，能被像本地内存一样直接寻址和拷贝**」。生态栈位对照：

```text
vllm-ascend MemcacheBackend ──对应──► vLLM MooncakeStoreConnector
memcache_hybrid.DistributedObjectStore ──对应──► Mooncake Store（对象/生命周期层）
MemFabric（BM 元数据 + TransferEngine + offload）──对应──► Mooncake Transfer Engine（传输/内存层）
昇腾驱动（devmm SVM/灵衢 UB/HCOMM）──对应──► NIC 驱动 + RoCE/IB
```

即 **MemFabric : MemCache : : Mooncake TE : Mooncake Store**——前者是内存/传输底座，后者才是 KV 对象管理层。

## 2. GVA 全局统一编址：寻址即算术

README 说「所有进程的 GVA 起始一致、线性排布一致」——实现本体是一个**不变式**（`hybm_mem_segment.h: FindRemoteRankByVa`）：

```text
gva = base + rank × maxSize + delta    →    rank = (gva − base) / maxSize   // 纯除法，零查表
```

所有 rank 预留 `rankCnt × maxSize` 的**对称窗口**，本地物理只映射自己的槽。对比普通做法（NCCL/自研 RDMA 每对 rank 交换 (addr, rkey, len) 元数据表 + 运行时查表寻址）：**一份 GVA 全集群语义一致，跨 rank 寻址不需要任何运行时元数据交换**。

实现细节（面试可引用的锚点）：
- 地址空间：GVM 常规池区从 40T+1G 到 128T（`0x280040000000` 起，"偏移 1G 规避底层问题"），**最前 2T 独占给 offload 场景**——与 BM/Trans 物理隔离；Trans 场景另有独立 56 位区（`1<<60~1<<61`）；
- 页表注入 = 对驱动 devmm_svm 子系统的 ioctl（`DEVMM_SVM_ALLOC/ADVISE`，介质由 advise 位区分 HBM vs DDR 巨页）+ **用户态重建驱动的虚拟堆管理**（`GvaHeap` 直接操作驱动共享的 `DevVirtHeapMgmt` 红黑树原语，实现指定 VA 点名分配）；GVA 版本 V1-V4 按 HDK 版本字符串分叉；
- 物理内存：DRAM 池用 `HalMemCreate(GIANT_PAGE_1GB)`，OOM 降级 2MB 大页；跨节点导出用 `HalMemExport/Import` 得 fabric 共享句柄；对齐要求 2MB（驱动层）/ GB（segment maxSize）/ 4K（device_rdma 注册 DRAM 首地址）。

## 3. xcopy 数据面：把 host 开销从 O(IO 数) 压到 O(1)

**one-copy 的准确含义**：GVA 直接填进 DMA 描述符（SDMA SQE / HCOMM 描述符 / RoCE WQE），DMA 引擎在本端与远端介质间直搬——host 只下发一次描述符。注意诚实的边界：SDMA 路径的 host 源/宿（LH↔G）实际仍是两跳（先 `AclrtMemcpy` 到临时 device buffer 再 G2G）；真一跳发生在 device 侧发起的 G2G 与 URMA/RDMA 路径。

**十引擎矩阵**（`HYBM_DOP_TYPE_*` 位掩码）：MTE（AI Core 算子内 UB↔GM 直访 GVA）、SDMA（超节点内 Die/片间 copy engine）、DEVICE_RDMA（RoCE QP）、DEVICE_URMA/UBOE（UB 网络单边）、HOST_RDMA/TCP/URMA、HOST_SHM（同节点 memfd）、AIV_SDMA、LD/ST。选择逻辑**不做运行时测速**——三级决定：用户位掩码 → tag 对规则（rank-pair 粒度）→ 固定优先级 **SDMA > DEVICE_RDMA > DEVICE_URMA > HOST_***（SDMA 可达性按 superPodId/serverId 拓扑判断），沿列表逐个 try。

**batch 设计是 KV 场景的针对性答卷**（README 时延测试就是 122 个离散地址模拟 DeepSeek-R1 KV block）：
- URMA 多 rank 批量：host 预构描述符数组 + per-rank marker → **一个 device kernel（`HybmBatchTransfer`，AICPU 上 dlopen `libccl_kernel.so` 与 HCCL 共用设备原语）下发最多 8190 个离散 IO**——描述符驻留 device 内存，kernel 内部循环执行，host 开销 O(1)；
- SDMA 批量：先**合并物理连续段**再逐段异步；extend 模式用 64MB 锁页参数区（CAS 抢 170 槽、并发 32）支撑 **一次 kernel 搬 16K 个离散段**；
- 最激进路径：**AIV 直接 RDMA**（`smem_shm_aicore_base_rdma.h`）——AI Core 上手工构造 RoCE WQE+SGE、按 hns_roce_v2 格式逐 64B cacheline 刷写并 ring doorbell、轮询 CQE owner bit——**host 网络栈从数据面完全剔除**；SDMA SQE 同样是用户态直组（绕过 CANN runtime stream 封装，每线程一条 STARS 流）。

## 4. 三大 API + offload：同一引擎的三种寻址模型

| API | 寻址模型 | 场景 |
|---|---|---|
| **BM**（Big Memory） | 对称 rank 窗口（`smem_bm_ptr_by_mem_type(memType, peerRank)` 返回 GVA） | 大容量 HBM+DRAM 池，L2G/G2L/H2G/G2H/G2G 拷贝，动态 join/leave/扩容 |
| **Trans** | **uniqueId（ip:port）点对点**，替代 rank 概念；注册用户 HBM（非对称 segment） | 推理 D2D 传输——vllm-ascend `batch_transfer_sync_read` 走的这层 |
| **SHM** | 卡侧对称槽（`gva + symmetric_size×i`），AICore 零成本寻址 | 算子场景：卡侧 MTE/LD-ST 直访 + host 侧 barrier/allgather 控制原语 |

**Acc Offload**（KV 卸载专用）：LOCAL（每 rank 独占 DRAM 池）vs SHARED（多 rank 共享全池 GVA，初始化时 **PoolFingerprint AllGather 交叉校验**各 rank 配置一致）。四个算子里最关键的是 `sparse_copy`——**src/dst/len/count 参数全部驻留 device 内存、kernel 自己读**，host 不 baking 任何标量 → 天然支持 stream/ACL graph 重放（这正是 vllm-ascend sparse decode offload 能进图捕获的原因）；另有 `kv_exchange_copy`（page_first↔layer_first 布局互换，42 个 meta 全在 device，token 索引不过 CPU）。

## 5. 控制面与 HA：薄到极致

一条 Host TCP 链（acc_links：16B 头 + epoll，body≤64MB）+ 一个 config store（TCP/etcd/外部三种后端，KV 命令集含 WATCH_RANK_STATE/CAS）承载全部成员关系——**内存池成员状态机（GVA 布局、MR、endpoint）放进 store，join 是 server 驱动的集群级协调事务**（AddToWhitelist → EstablishConnection → PromoteToActive 三段命令序列）。

HA 设计值得单独记：etcd **只用于 leader 选举**（Lease TTL 5s + 分布式锁；元数据本体仍在 leader 的 TCP store 内——etcd 语义面小）；防脑裂的关键动作是 **leader 检测到 etcd 失联（≤4s）主动 StopServer 并断开全部 follower**——「旧 leader 必先降级再有新 leader」，对照 Mooncake 的 kLeaderWarmup（等旧租约过期）是两种防脑裂路径：**主动退位 vs 等任期**。

## 6. QoS、可观测与限制

- **QoS 是建链属性而非 per-IO**：`MF_DEVICE_UB_QOS`（0-7）在 Prepare 建 channel 时写入 HCOMM channel 描述符；RDMA 走 `MF_DEVICE_RDMA_TC/SL`（RoCE 差分服务标记）；改优先级需重建 channel——对照 Mooncake 的 TENT QoS（slice 级动态）粒度更粗；
- **ptracer**：上百个「方向×引擎×阶段」tracepoint 常开（Release 不摘），P50/P99/P999 分位直接落盘——生产可拿到「哪条引擎哪个阶段慢」的证据而非只有日志；故障注入四动作（callback/pause/reset/abort）+ 配置文件热激活；
- **性能锚点**（A3 超节点 UB 1.0，单 DIE+单 CPU）：RH2D 110 GB/s、D2RH 74.5 GB/s、RD2D 166 GB/s、D2RD 138 GB/s；时延测试条件即 KV 场景（8.57MB/122 离散地址）；
- **关键限制**：A2+HCOMM 路径的 channel MR 表是建链快照——join 后动态注册内存（vLLM 动态 KV cache 正是这种）不能直传，需走 swap 中转（故 `MF_HYBM_RDMA_USE_HCOMM` 默认 0，用 native RDMA 实时查 MR）；HBM 池化 8 个方向全部不支持 host_rdma；worldSize ≤1024；BM HBM/rank ≤64GB；深度绑定昇腾驱动（HDK 版本分叉 V1-V4）。

## 7. 与其他 KV Cache 管理组件的对比

| 维度 | MemFabric(+MemCache) | Mooncake (TE+Store) | LMCache | SGLang HiCache L3 |
|---|---|---|---|---|
| 层次 | 内存语义层（xcopy+GVA），对象语义在上面单独的 MemCache | 传输层（TE）+ 对象层（Store）一体 | 引擎侧缓存管理框架 | 引擎内置 L3 抽象 |
| 寻址模型 | **GVA 算术**（除法得 rank，零元数据交换） | TransferRequest (src/tgt/offset/len) 逐次寻址 | chunk key 查 StorageManager | page 级 + 后端 batch_exists |
| 数据面 | 10 引擎矩阵；batch 描述符驻留 device，host O(1)；AIV 直写 WQE | 自研 TE：拓扑矩阵+多 NIC slice spraying+failover | 委托 NIXL/UCX/GDS 等后端 | 后端自带（Mooncake/NIXL/...） |
| 缓存语义 | **无**（无对象/驱逐/租约/前缀） | 完整（lease/pin/quota/副本/SSD 分层） | 完整（chunk/prefix/pin/配额） | 树内元数据+按需查询 |
| 控制面 | 薄 TCP+KV store；etcd 仅选主；**leader 主动退位防脑裂** | Master 集中分配+OpLog/快照 HA；kLeaderWarmup 等任期 | Coordinator 事件投影 | 无中心（按需查） |
| 平台 | 昇腾深度绑定（devmm ioctl/灵衢/HCOMM，HDK 版本分叉） | NVIDIA/Ascend 多传输可插拔 | CUDA/ROCm/XPU/HPU/MUSA | 跟随 SGLang |
| 适用边界 | A3/A2 超节点内最优（SDMA 166GB/s）；跨平台为零 | 集群级、多厂商通用 | 多引擎、多后端插件化 | 引擎内一体化 |

**三家哲学一句话**：Mooncake 把复杂度花在**拓扑与对象生命周期**（网络异构性是它的敌人）；MemFabric 把复杂度花在**寻址与数据面**（host 开销与元数据交换是它的敌人），代价是平台绑定；LMCache 把复杂度花在**引擎适配与语义**（layout 多样性是它的敌人）。选型上它们不互斥：vllm-ascend 的栈里 MemFabric 做底座、MemCache 做对象层、AscendStoreConnector 做接入——**同一个生态位（NPU 上的 KV 池化）用华为全家桶替代了 Mooncake 的位置**，这正是「Mooncake 在昇腾上的镜像」。

> 🎯 **面试考察点**：NPU 岗的高分题「GVA 怎么实现全局统一编址」（答对称窗口+除法算术，对比元数据表交换）；「one-copy 是什么」（GVA 进 DMA 描述符 + 8190 desc 一个 kernel + host O(1)）；「昇腾 KV 池化与 Mooncake 路线的区别」（层位对照表）。相关：[SGLang 与 vLLM-Ascend 池化设计](/ecosystem/sglang-ascend-pooling)、[H2D/D2H 数据搬运对比](/core/h2d-d2h-deepdive)。
