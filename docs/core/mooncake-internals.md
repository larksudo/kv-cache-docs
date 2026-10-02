---
prev: /core/lmcache-internals
next: /ecosystem/production-claims
---

# Mooncake 内部机制深潜

> 与「LMCache 内部机制深潜」对称：把 Mooncake 的 Master 元数据、OpLog/快照、Count-Min Sketch、故障体系、SSD 分层、EP/PG 弹性、P2P 权重分发拆到函数/参数级。前置阅读：「可靠性与数据库视角」（协议概念）与「存储引擎中的数据变形」（对象结构）。
>
> 来源：「一研」Mooncake 系列 31 篇（本地快照 `references/raw/yiyan/`），关键断言已对照 Mooncake 源码验证（1024 分片 ✓、CMS 16KB ✓、fork 快照 ✓、OpType 枚举 ✓）。注意行号引用基于文章成文时版本，本地快照的 OpLog 已重构为 `ordered_oplog_writer.h` 等文件（机制一致、API 名有差异）。

## 1. Master 元数据：1024 分片 + 三级锁

### 1.1 分片结构

```cpp
// mooncake-store/include/master_service.h
static constexpr size_t kNumShards = 1024;
struct MetadataShard {
    mutable SharedMutex mutex;                          // 读写锁
    std::unordered_map<TenantId, TenantState> tenants;  // 租户表
    long disk_object_count;  // 有 COMPLETE LOCAL_DISK 副本的对象数
};
std::array<MetadataShard, kNumShards> metadata_shards_;
```

为什么是 1024：百万级对象时每分片约 1000 个对象、锁竞争小；2 的幂可用位运算取模。`disk_object_count` 是容易被忽略的关键字段——驱逐基数 `eviction_base = metadata.size() - disk_object_count`，磁盘-only 对象不占内存、不参与内存驱逐。

**分片路由差异化**（`getShardIndex`）：default 租户直接 `hash(user_key) % 1024`；非 default 租户用 `boost::hash_combine(tenant_hash, user_key)`——防止不同租户的相同 key 挤到同一分片。

**独立的配额分片**：`kNumTenantQuotaShards = 1024`，与元数据分片完全分离——PutStart（写配额）不阻塞 GetReplicaList（只读元数据）。`TenantQuotaState` 生命周期：PutStart `reserved += size` → PutEnd `used += size, reserved -= size` → Remove `used -= size`。

### 1.2 三级锁层次

获取顺序（全局注释约定）：`client_mutex_ → tenant_quota_policy_mutex_ → snapshot_mutex_ → metadata_shards_[i].mutex → tenant_quota_shards_[...].mutex → segment_mutex_`。

| 级 | 锁 | 保护什么 |
|---|---|---|
| L0 | `ObjectOperationLock`（4096 条纹 mutex） | 同一 key 的 PutStart/PutEnd/Remove 串行化 |
| L1 | 分片 SharedMutex | 每分片的租户表 |
| L2 | `ObjectMetadata::lock`（SpinLock） | 租约字段 |

### 1.3 ObjectMetadata 与副本

字段三类：**身份**（不可变：client_id/size/group_id/tenant_id/user_key…）、**生命周期**（lease_timeout 硬租约、soft_pin_timeout、hard_pinned const）、**配额记账**（reserved/committed/pending 三态）。delete 拷贝/移动构造 → 内存位置唯一防悬空指针；构造/析构自动增减指标。

副本状态机 `UNDEFINED → PROCESSING → COMPLETE → REMOVED/FAILED`；副本类型 `MEMORY / DISK / LOCAL_DISK / NOF_SSD`，数据载体是 `std::variant`。副本 ID 全局原子递增从 1 开始——同一 key 删除重建后新 ID 必更大（版本语义）。

**租约三道保护与半程刷新**：硬钉扎 > 软钉扎 > 硬租约 > 无保护（驱逐优先级相反）。`GrantLease` 只延长不缩短；`NeedsLeaseRefresh` 在 `lease_timeout <= now + ttl/2` 时触发——半程提前刷新避免「到期→驱逐→重分配」竞态。同 group_id 的多对象合并续约（一个 prompt 的多层 KV 同 group），减少 etcd 写放大。

## 2. 持久化：双路径 OpLog + Fork 快照

### 2.1 OpLog

`OpType`：PUT_END=1 / PUT_REVOKE=2 / REMOVE=3（**LEASE_RENEW 已废弃**——租约续约高频，每条记 OpLog 写放大严重；Standby 只需靠 DELETE 感知数据消失）。

**双路径**：`Append()` 异步 best-effort（仅 PUT_END，微秒级）vs `AppendAndPersist()` 同步等 etcd（REMOVE/PUT_REVOKE，毫秒级）。判断逻辑一行：`bool sync = (type != OpType::PUT_END)`。同步的原因：**Standby 提升前必须观察到 DELETE，否则可能返回指向被复用内存的陈旧描述符——静默数据损坏**。

工程细节：序列号内存预分配（etcd 写失败也不复用，重试用相同序列号）；内存缓冲有界 deque（10 万条、key≤4KB、payload≤10MB——DoS 防护）；etcd 批量 Group Commit（100 条或 1s 触发）；**etcd key 20 位零填充**（`kOpLogBatchIdWidth = 20`）使字典序=时间序，支撑范围删除。

### 2.2 Fork 快照

主进程 `fork()` → 子进程获得 COW 内存快照 → 逐分片序列化（msgpack）→ Zstd level 3 压缩 → 上传；**主进程不阻塞**（百万级对象在线序列化需长时间持读锁，fork 是唯一不阻塞方案）。存储双分离：catalog store（latest/manifest/descriptor）vs object store（数据文件）。快照是周期性的——最后一次快照后的变更靠 OpLog 重放补齐。

### 2.3 恢复与 HA 状态机

恢复三步：加载快照（基线）→ 重放 OpLog（seq > snapshot_sequence_id）→ Watch etcd 前缀实时同步。**OpLog 缺口处理**：PUT_END 丢弃（数据可能不完整，宁可丢不可错）；REMOVE/PUT_REVOKE 应用（幂等）。`OpLogApplier` 的 gap 策略：1s 窗口内先缓存、主动向 Primary 请求缺失、3s 后按上述规则跳过。

Master 状态机：kStarting → kStandby → kCandidate → kRecovering → kCatchingUp → **kLeaderWarmup**（当选后等旧 Leader 的 etcd 租约过期，防脑裂）→ kServing。边界结论：**Master 崩溃时客户端不能读**——即使数据还在，没有 Master 就无法定位 segment/offset；etcd HA 下故障窗口是秒级而非分钟级。

## 3. Count-Min Sketch：16KB 的两个岗位

```cpp
// count_min_sketch.h: width 4096 × depth 4 × uint8 = 16KB
// increment(): 4 行独立哈希递增，返回最小值（只高估不低估——"安全的错误"）
// decay: total_increments >= 16384 时全体 >>= 1（单周期右移，不用除法）
```

为什么不用精确计数：100 万对象精确计数需 100-200MB 且衰减需逐条遍历；CMS 衰减是 16KB 网格一次右移。uint8 足够——淘汰只需区分 0/1/≥2 三档。

两个使用场景（关键增量）：

1. **Client 端本地热缓存准入**（「二刷门卫」）：`ShouldAdmitToHotCache(key, cache_used)`——缓存命中**不递增**（防「富者愈富」正反馈），未命中才递增；`admission_threshold_ = 2`（被访问 ≥2 次才值得进本地热缓存）；
2. **Master 端 SSD→DRAM 提升的四道闸门**：频率（CMS ≥ threshold）→ 水位（DRAM < 高水位）→ 去重（无在途任务且无 MEMORY 副本）→ 容量（提升队列 < 50000）。

## 4. 故障体系：检测→隔离→降级→恢复

**三层心跳检测**：Store 层 Client→Master Ping 1s（连续 3 次失败切 Leader）；PG 层 ConnectionPoller 8ms→1024ms 指数退避；传输层 RDMA QP 硬件重试（timeout=14, retry=7）。

**O(1) 隔离**：PG 的 `activeRanks` 放在 **cudaHostAlloc 页锁定内存**——CPU/GPU 都可直接访问，检测到断连立即翻转标志位，下一次 All-to-All 自动跳过；传输层 `Topology::disableDevice()` 从所有拓扑条目移除故障网卡。

**降级选 AP**：GPU rank 故障跳过专家（质量略降不丢请求）；网卡故障用剩余网卡；Master 故障 Standby 接管。

**两阶段销毁**（防 use-after-free）：RDMA 端点 `beginDestroy()`（QP 置 ERR 让硬件自动刷在途 WR）→ `finishDestroy()`（`wr_depth_list_` 全归零才真正销毁；30s×3 次兜底）。段卸载 `GracefulUnmountSegment`：宽限期内 **TE MR 保持注册**（远端 peer 仍可读），到期才注销。

Master 主备切换六步：etcd lease 过期 → `CreateWithLease` 原子选举 → OpLog 追赶 → `PromoteStandby()` → 启动 RPC + **设置 Pod Label `mooncake.io/store-role=leader`**（Service selector 跟随，Client 无需知道 Leader 是谁）→ Client 自动重连。

## 5. SSD 分层的工程细节

四副本类型延迟谱：MEMORY ~100ns / LOCAL_DISK ~10μs / NOF_SSD ~20μs / DISK ms 级——**同一对象可同时拥有 MEMORY+LOCAL_DISK 副本**（热货架+冷库各一份）。

- **Offload 流程**：Master 水位决策 → Client 心跳拉任务 → `BatchQuerySegmentSlices` 取切片 → GPU 数据经 PinnedBufferPool D2H 暂存（DRAM 数据直通用 `FindDeviceForPointer` 自动判断）→ `BatchOffload` → 注册 LOCAL_DISK 副本；
- **Promotion 尽力而为**：任何一步失败跳过，Master reaper 在 TTL 后清残留——「宁可不做，不可做错」；
- **BucketReadGuard 非对称内存序**：读计数递增用 `memory_order_relaxed`（近似即可）、递减用 `memory_order_release`（必须保证读操作对驱逐线程可见）——一行代码的并发纪律；
- **双分配器哲学**：CacheLib slab（分配极快但无法精确报告空闲，已弃用）vs Offset 伙伴系统（256 大小类、精确统计、可 serialize 用于恢复——默认）；
- **AllocationStrategy 五种**：Random（LLM 对象均匀时反而优于 free_ratio_first——省计算）/ FreeRatioFirst（**采样 6N 候选**按空闲比排序选 Top-N，失败回退随机——Power of Two Choices）/ SsdFreeRatioFirst / Cxl / local_first；
- Master 重启恢复：心跳返回 `SEGMENT_NOT_FOUND` → 自动重挂载段 + 异步 ScanMeta 重建 SSD 对象元数据。

## 6. EP 与 PG：MoE 的弹性底座

### 6.1 Expert Parallel（遵循 DeepEP 低延迟模型 + 故障容忍）

- **拓扑发现**：节点内 NVLink（~160GB/s）+ 跨节点 IBGDA/RDMA（~50GB/s）混合；选路 `同 scaleout → NVLink else RDMA`。注意边界是**物理服务器**而非 NUMA 节点（HGX H100 是 8×8 NVLink mesh）；
- **Buffer GPU 布局**：RDMA 信号区（每远端 rank 一槽）+ 数据区 + NVLink/IPC 区 + CUDA 计数器；**双缓冲**——Buffer 0 做 Combine 时 Buffer 1 已开始下一轮 Dispatch；
- **CUDA kernel 内超时检测**：`timeout_ticks` 在 kernel 内部判断，已就绪 rank 立即计算不等超时者——故障 rank 的剔除不需要 CPU 介入；
- **零拷贝 Combine 语义澄清**：省的是 GPU 显存一次本地拷贝（专家直接 `out=buffer` 写最终位置），**不是省通信**——All-to-All 回传省不掉；
- 行业数据：DeepSeek-V3 671B 总参/37B 激活（5.5%）、256+1 专家 K=8；58 层 MoE 每次前向 dispatch+combine 约 5.8GB；跨节点 EP 通信占总推理 30-50%；NVLink 单层 All-to-All ~0.5ms vs IB ~1.7ms vs TCP ~8.7ms；**Wide EP**（EP=64 每 GPU 4 专家）把计算延迟 5ms→0.6ms——LMSYS 128×H200 跑 Kimi K2 达 2.2k tok/s/H200；FP8 收益 MoE +69% vs 稠密 +31%（MoE 计算通信比偏通信）。

### 6.2 Process Group：缺席跳过的集合通信

注册为 PyTorch c10d 后端 `mooncake`/`mooncake-cpu`——NCCL/Gloo/MPI 都不能跳过缺席者，这个后端可以。核心概念：size（含故障槽位）/ activeSize / `activeRanks_` / `maxWorldSize_`（官方建议 ≥ world_size × 1.5 预留替补）。

**弹性恢复三阶段**：替补 rank 以 `is_extension=True` 加入并发布本地元数据等待 → 健康 rank `get_peer_state()` 发现 → `recover_ranks([new])` 激活 → 替补 `join_group()` 返回参与集合通信。与 EP 协作（SGLang）：`recover_ranks` → `ep_buffer.update_ep_member()`。

### 6.3 P2P Store：模型权重分发（「一火传千灯」）

定位与 KV Cache 无关——**模型权重/checkpoint 分发**。无 Master，etcd 做元数据火种簿；Go 实现。`Register` 只登记元数据不传数据；`GetReplica` 查 etcd 选火源 → TE 拉取 → **CAS 更新 etcd 登记自己为新传火者**（`Compare(ModRevision)` + OpPut，revision 不匹配重试）——参与者越多分发越快。

与 direct PD 的代码级对比：元数据 etcd（1-5ms、CAS 乐观并发、持久化）vs ZMQ（0.1ms、无并发控制、临时）；段发现 etcd 查询 vs P2PHANDSHAKE 直连；**传输方向消费者拉（READ）vs 生产者推（WRITE）**——PD 选推送因为只有 P 知道数据就绪；PD 独有 `group_consecutive_contiguous()` 连续块合并。实测：Kimi-K2（1T 参数约 2TB）分发到 1000 GPU——Mooncake Store 估算 5-10 分钟 vs P2P 实际约 20 秒；SGLang RL 场景 53s → 7.2s。选择：权重分发/热更新/冷启动 → P2P；实时 KV/前缀缓存/多租户 → Store；两者可同集群共存。

## 7. 硬件交互速查

| 层 | 延迟 | 带宽 | 容量 | 成本 |
|---|---|---|---|---|
| HBM3e | ~100ns | ~2TB/s | 80-192GB | ~$20/GB |
| DDR5 | ~100ns | ~50GB/s | 256GB-2TB | ~$5/GB |
| NVMe | ~10μs | ~7GB/s | 3.84-15TB | ~$0.3/GB |
| NVMe-oF | ~20μs | ~6GB/s | 集群级 | ~$0.1/GB |

- **DRAM 跨进程共享**：`memfd_create` → `ftruncate` → `mmap(MAP_SHARED|MAP_POPULATE)`，fd 经 Unix Socket 传递；2MB 大页已成 KV Cache 系统业界标配（4KB 页注册 1GB MR 需数秒，2MB 页几十毫秒）；
- **D2H**：Pageable ~1GB/s vs Pinned ~25GB/s（PCIe 4.0）——**25 倍差距**；PinnedBufferPool（32 缓冲）Acquire 优先取池、平台适配、优雅降级（失败回退 new char[]）；大块注册 1-2TB 需 3-5 分钟 → 8 线程分批；
- **io_uring**：进程级共享 ring 单例 + 预注册固定缓冲区 + `batch_read()` 一次最多 32 个独立读；**BatchLoad 零拷贝技巧**：数据可能从非对齐偏移开始——对齐缓冲区读入后调整指针跳过填充，无额外 memcpy；
- **昇腾独有**：`ascend_allocate_vmm_memory_direct()` 直接分配 fabric memory（非 memfd 路径）。

## 8. 四个工程技巧的实现细节（一研「会思考的数据高速公路」篇）

1. **多通道切片传输**：`RdmaTransport::submitTransferTask()` 按 slice_size（默认 64KB）切，每切片独立 `selectDevice()` 选网卡；**尾部合并**——剩余 ≤ slice_size + fragment_size 时并入最后切片，避免微小尾部 RDMA 写；
2. **SIEVE 端点池化**（NSDI'24）：每端点一个 visited 位；命中置 true；淘汰从尾部 hand 扫描——visited=true 清零放行、false 淘汰；新端点插头部。**快速降级语义**：第一次扫描就清零，不给第二次机会——必须两次扫描间被再次访问才能存活。查找路径 = 一次哈希 + 一次 atomic_bool，比 LRU 少链表操作和写屏障；
3. **RDMA 建连开销分解**（为什么必须池化）：创建 QP×2 数百微秒 + TCP RPC 握手毫秒级 + QP 状态机 8 次内核转换（RESET→INIT→RTR→RTS × 2 QP）+ 对端同样——**总计 1-10ms，而 Decode 步骤本身可能只几毫秒**；
4. **跨 NUMA 实测**：NIC1→NIC4 同 NUMA 21.9GB/s / 5.67μs；NIC1→NIC6 跨 NUMA 18.9GB/s / 8.30μs（延迟 +47%、带宽 -14%）。

**KV Cache 尺寸速查表**（一研「元数据与数据流」篇，规划传输带宽用）：

| 模型 | 4K | 32K | 128K |
|---|---|---|---|
| LLaMA-70B (GQA-8, tp=8) | 160MB | 1.25GB | 5GB |
| GLM-5 (MLA, tp=8) | 437MB | 3.4GB | 13.7GB |
| Kimi-K2.6 (MLA, tp=8) | 274MB | 2.2GB | 8.7GB |

16GB 的请求若走代理模式能把 Master 的 100Gbps 网卡堵 1.3 秒——这是「Master 不碰数据」的动机。读取无 "GetEnd"（不修改元数据状态）；部分成功由 `DetermineFinalizeDecision` 决定哪些副本 PutEnd、哪些 PutRevoke，至少 1 副本成功则整体成功。

**Store 客户端 API 补遗**：PyClient 全集 `put/get_into/remove/put_batch/batch_get_into/batch_remove/create_copy_task/create_move_task/query_task/health_check`；`ReplicateConfig{replica_count/preferred_segments/soft_pin/hard_pin}`；三种部署模式 Embedded / Dummy-Real / Standalone-store。

## 9. 18 问精选（一研「架构、传输与容错」篇）

- **传输慢 + CPU 占用高 = 走了 TCP**（非零拷贝的标志）；
- 64MB 单网卡速率 → 多通道切片传输未配置；每次传输建拆 RDMA 连接 → SIEVE 端点池化没生效；
- 新硬件加速器通过 **installTransport 可插拔**（策略模式）扩展 TE；
- preferred_hca（最快）/ avail_hca（较慢备用）：网卡故障自动回退 avail 重试；install() 需要拓扑参数否则无备用路径；
- MoE 推理 GPU 崩溃防挂起 = active_ranks 掩码；
- **Master 崩溃时客户端不能读**（需 Master 定位 segment/offset，数据在也找不到）；etcd HA 下故障窗口秒级而非分钟级；
- Master 从 fork 快照恢复不丢数据的原因：**数据从未离开 Client——Master 只存元数据**。

## 10. 部署形态速览

库优先、进程级——「提供积木不提供蓝图」。Master 镜像（`docker.io/kvcacheai/mooncake`）**不含真实 GPU 运行时**（stub libcuda 满足链接器，GPU 只在 Client 侧）。K8s：Lease 选举（5s/3s/1s，ReleaseOnCancel）+ Pod Label 服务发现 + 三重资源亲和（GPU+RDMA+HugePage 须同 NUMA）。四层配置优先级：gflags > YAML > `MOONCAKE_*`（高层）> `MC_*`（底层调优）。健康端点 `/health` `/role` `/ha_status` `/metrics`。

测试规模：1,908 个 gtest 用例（store 1298 / TE 279 / TENT 291），24 个 CI 工作流；六个测试技巧中两个值得借鉴——**FaultProxyTransport**（传输装饰器注入 `submit_fail_rate/status_corrupt_rate` 等故障率，可让 RDMA 100% 假失败验证 TCP 故障转移）与 **--wrap 链接器符号替换**（拦截 ibv_* 符号，无物理网卡的 CI 也能测 RDMA 故障恢复）；**mooncake-p2p-store 零测试**是选型时的已知风险。
