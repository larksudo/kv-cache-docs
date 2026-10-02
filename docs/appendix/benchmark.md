---
prev: /ecosystem/selection
next: /appendix/evidence
---

# 验证方法与指标

如何公平地 benchmark KV cache 系统？本篇保留 V5 中方法论价值最高的三部分：冷热状态控制、流量模型、常见失败模式解读，以及指标定义。

> 部署命令清单（版本采集、硬件拓扑采集、逐场景压测步骤）已按「技术分析为主」原则从本知识库移除；需要时可参考原 V5 文档或各系统官方 quickstart。

## 5. 公平 benchmark 方法

### 5.1 冷热状态

必须分别报告：

1. cold：清空 cache、清 page cache、重启 server；  
2. warm：预写目标 prefix，但不重复请求；  
3. hot：连续命中；  
4. partial：前 50%/80% 命中；  
5. eviction pressure：池满且持续写入。

只报 hot/warm 数据会造成严重误判。

> **官方 MI300X 基准的方法论教训（2026-09 补充）**：固定命中率的合成基准里 LMCache 吞吐反而低 10-17%（HBM 无压力时 L2 纯付 connector 开销），真实 agent trace 高压下 +200%——**公平的 L2 基准必须同时具备「复用」和「HBM 压力」两个条件**；且存在 regime crossover（工作集 <100k token 用 HBM prefix cache 即可，250k+ 才到 LMCache DRAM 主场），报告数字必须附工作集规模。

### 5.2 流量模型

| 模型 | 目的 |
|---|---|
| single turn | 排除复用收益 |
| multi turn | 测增量上下文 |
| shared system prompt | 测公共前缀 |
| agent tool trace | 测长尾 fragment |
| random prefix | 测索引/池开销 |
| burst same prefix | 测缓存击穿 |
| poison-like overlap | 测 key/salt 隔离 |

### 5.3 指标归档

每个结果必须保存：

```text
git commits
image digests
config files
hardware topology
request trace hash
random seed
SLO 定义
cache state
error logs
prometheus snapshot
per-request CSV
```

---

## 6. 常见失败模式

| 现象 | 可能原因 |
|---|---|
| TTFT 改善但 TBT P99 变差 | H2D 抢占 PCIe、cache 线程抢 CPU、GPU block pin 过久 |
| hit rate 高但吞吐不升 | load 串行化、block allocation 等待、CPU deserialization |
| RDMA 带宽高但端到端无收益 | 瓶颈在 prefill/Master/lock，不在传输 |
| cache 服务正常但业务无收益 | routing 不命中、cache salt/tenant key 不一致 |
| 长压测后变慢 | 对象/IPC handle/rkey/endpoint 泄漏 |
| failover 后数据错误 | block pin/lease/job ledger 不完整 |
| SSD 冷读有效但热路径变差 | SSD restore 占用 staging buffer 或 NIC |
| Master failover 后部分 key 消失 | snapshot/OpLog RPO 或 client heartbeat 清理 |

---

## 后续可验证工作（主报告第 13 章）

### 13.1 端到端实验矩阵

| 实验 | 目标 |
|---|---|
| vLLM + LMCache LocalCPU + L2 disk/GDS | 单机 multi-turn TTFT/吞吐 |
| vLLM + LMCache MP CUDA IPC vs SHM | 进程边界拷贝成本 |
| vLLM + LMCache NIXL RDMA 1P1D/xPyD | PD 直传与 chunk reservation 稳定性 |
| vLLM + LMCache P2P CPU RDMA | peer failure、lock、驱逐、部分命中 |
| vLLM + Mooncake Store embedded/standalone | 全局池命中率和 Master 压力 |
| vLLM + Mooncake direct PD | 异构 TP/PP、region align、transport fallback |
| vLLM + MultiConnector(PD + Store) | 直传+池化组合是否提升 SLO goodput |
| SSD offload 压测 | 冷读、SSD->GPU、写放大、CRC/recovery |
| 故障注入 | worker OOM、peer 掉线、NIC fail、Master failover、coordinator 重启 |

### 13.2 指标

性能：

- TTFT/TBT/E2E；
- prefill compute saved；
- external hit rate；
- store/load bytes/s；
- DMA CPU 占用；
- GPU SM 占用；
- P99 store latency；
- block pin 时间；
- request queueing delay。

可靠性：

- worker/daemon crash 后 cache 命中；
- NIC/Master failover 时间；
- partial store 清理；
- lease expiry 造成的读失败；
- coordinator 重启后目录收敛时间；
- 对象/块泄漏。

### 13.3 源码级审计清单

1. LMCache MP 协议版本兼容矩阵；  
2. LMCache Mooncake L2 adapter 的 key/quota/eviction 映射；  
3. Mooncake vLLM Store worker 的 partial failure 回滚；  
4. Mooncake TE submit-stage partial enqueue 的影响；  
5. HMA/hybrid model 下两个 connector 的支持边界；  
6. KV event 在多 worker/多 connector 下的去重与一致性；  
7. GPU block pin 到 release 的最长路径；  
8. cache salt/tenant 隔离是否能阻止跨租户 key 冲突；  
9. SSD checkpoint 后 stale seq/CRC 行为；  
10. RDMA memory registration 与 hugepage/NUMA 配置。

---

---

## 附：最小验收清单（原 V5 第 7 章）

### 7.1 LMCache

1. LocalCPU 命中收益 >2x TTFT，无命中开销 <2%；  
2. MP worker OOM 后缓存可继续服务；  
3. P2P peer kill 只产生 miss；  
4. async PD 在 8 并发 chunked 请求下无死锁；  
5. L2 store failure 不损坏 L1；  
6. coordinator 重启后 quota/directory 可收敛；  
7. Prometheus/Grafana 能区分 L1/L2/P2P 命中。

### 7.2 Mooncake

1. TE multi-NIC 吞吐接近单 NIC 线性；  
2. NIC failover 后任务不返回错误数据；  
3. Store Put/Get 无 dirty read；  
4. Master failover 后新 leader 通过 promotion validation；  
5. lease expiry 时读失败可被 vLLM 重试/重算；  
6. store job 完成 后 GPU block 引用释放；  
7. direct PD 异构 TP/PP region 校验通过；  
8. SSD offload 有 CRC/restart recovery 证据。

### 7.3 vLLM

1. connector 版本升级矩阵通过；  
2. MultiConnector PD+Store 无重复写导致数据错误；  
3. KV event 与实际 block store 一致；  
4. preemption 后 `requires_kv_delivery` 行为符合预期；  
5. HMA/hybrid 模型仅在支持矩阵内启用。

## 附：Mooncake 性能调优速查与踩坑（一研「性能调优/踩坑实录」篇）

**五道闸门**——RDMA 参数（默认→高吞吐/低延迟两档）：`MC_SLICE_SIZE` 64KB→256KB/32KB、`MC_NUM_QP_PER_EP` 2→4、`MC_MAX_WR` 256→512-1024、`MC_MAX_CQE_PER_CTX` 4096→8192、`MC_WORKERS_PER_CTX` 2→4-8、以太网 MTU 1500→**9000**（两层 MTU：MC_MTU 恒 4096 是 IB 规范上限，RoCE 下以太网必须 Jumbo 否则 RDMA 包被拆 3 帧）、`MC_IB_PCI_RELAXED_ORDERING` 0→1、QoS `MC_IB_TC` 与训练流量隔离。默认 2 QP×256 WR=512 在途，调优 2048 在途吞吐 2-4×。

其余四道：拓扑路由（`MC_ENABLE_HCA_PEER_AFFINITY` 与 `DEST_DEVICE_AFFINITY` **互斥只能开一个，推荐先试前者**）；内存（`MC_STORE_USE_HUGEPAGE`、GLOBAL_SEGMENT_SIZE 建议物理内存 50-70%、`MC_ENABLE_PARALLEL_REG_MR=1`——1GB MR 注册数秒、1TB 级 3-5 分钟）；SSD（`MOONCAKE_OFFLOAD_USE_URING` **最简单最有效 +30-50%**；心跳 10s→5s）；水位联动（高水位 0.95→0.85-0.90、`offload_on_evict` true、`promotion_max_per_heartbeat` 1→3-5）。

**P0 检查（不调白调）**：协议=RDMA（TCP 差 5-10×）、NIC 亲和、以太网 MTU 9000、io_uring——三步（RDMA+亲和+URING）解决 80% 问题。

**八个实战坑**：①RDMA 初始化失败静默回退 TCP（查日志 "fall back to TCP"）；②etcd 连接串必须 `etcd://` 前缀；③GPUDirect 需 `modprobe nvidia-peermem`，不支持则注册位置改 dram；④**Master 分配策略选错致 OOM**——对象大小均匀（LLM）用 random，差异大用 free_ratio_first，SSD 分层用 ssd_free_ratio_first；⑤容器内多网卡拓扑发现失败——需 `-v /sys:/sys`，`engine.showLinks(true)` 验证 preferred_hca 非空（4×200G 只有 25GB/s 即为征兆）；⑥QP 耗尽——`MC_MAX_EP_PER_CTX` 默认 65536，千节点集群需 131072；⑦容器 RDMA 五件套 `--device /dev/infiniband --cap-add IPC_LOCK --cap-add NET_ADMIN -v /sys:/sys --network host`；⑧排障参数 `MC_GID_INDEX`（RoCEv2 常用 3）/`MC_IB_PORT`/`MC_TRANSFER_TIMEOUT`。
