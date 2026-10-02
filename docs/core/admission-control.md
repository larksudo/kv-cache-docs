---
prev: /core/async-protection
next: /core/p2p-pd
---

# 准入控制与防挤占：有限资源的仲裁哲学

> 「机制对比」系列第三篇。命题：**staging 缓冲/host 内存/池容量是有限的，并发请求都想要**。没有仲裁就是三种死法——**死锁**（各持一半谁也完不成）、**抖动**（预取挤掉热数据）、**饿死**（低频请求永远排不上）。LMCache、SGLang、Mooncake 各自押了一个方向，三家放在一起就是 admission control 的完整光谱。

## 1. LMCache PD：事务式准入（correctness-first）

**死锁场景**（设计文档原文）：chunked prefill 下 receiver staging buffer = 10 chunks，请求 A、B 各需要 8：A 先到占 5，B 到占 5——**谁也拿不到剩余 3，谁也无法完成，谁也不释放**。这不是性能问题是正确性问题：系统在高压下不是变慢而是挂死。

**解法：all-or-nothing 预留**。`ReservationManager` 让 receiver 在**首个 batch** 就为整个逻辑请求预留 `total_chunks`；空间不足则整个请求等待，**绝不部分进入**；后续 batch 保证可分配；abort/失败全量回滚；`ProxyNotif` 要求 `completed_chunks == total_chunks` 才放行。工程佐证其严肃性：测试代码量（1081 行）接近实现（1663 行）——作者把 admission/failure/abort 当核心语义而非性能优化。

数据库类比：为多页写入预留 extent/redo 空间。**局部成功没有意义，逻辑单元必须整体能完成**。

## 2. SGLang HiCache：反馈式精确分配（waste-minimizing）

SGLang 面对的是另一个敌人：**抖动**。L3 预取如果激进预留 host 内存，会把 L2 热数据挤出去（预取的还没用上，在用的先被赶走）。

**解法：「命中数确定后才分配，绝不预分配」**（`_drain_and_alloc_storage_hit` 源码注释原文："this is the whole point: no over-allocation up front"）。流程：先 `batch_exists` 查 L3 实际命中页数 → **只为确认命中的页分配 host 内存** → 再发起 I/O。配套三件：

- `prefetch_tokens_occupied` 占用率计数 + `prefetch_rate_limited()` 总闸门（防预取洪峰）；
- `load_back` 全有或全无（< 10 token 放弃——太碎不值得搬）；
- **优雅降级链**（资源不足时不硬等也不失败）：命中数 < threshold（256 token）→ revoke；host 分配失败 → 先 `evict_host` 再试 → 仍失败 → **截断到页对齐前缀**（仍须 ≥ threshold）→ 才 revoke。

## 3. Mooncake：服务端治理（multi-tenant fairness）

Mooncake 面对的敌人是**饿死与公平**：多租户共享池，谁的热数据该被提升到 DRAM、谁的写入该被接纳？

**四道闸门**（SSD→DRAM 提升，按序检查）：① 频率——CMS 计数 ≥ 2（「二刷」才值得提升，且缓存命中不递增防「富者愈富」正反馈）；② 水位——DRAM 使用率低于高水位（0.95，建议调 0.85-0.90）；③ 去重——无在途任务且无 MEMORY 副本；④ 容量——提升队列 < 50,000。

**配额三态**：tenant quota 的 `requested → reserved（PutStart）→ committed（PutEnd）` 生命周期，超配额触发 tenant-scoped 驱逐；驱逐基数排除磁盘-only 对象（`disk_object_count`）。另外 PDBackend 的 alloc 循环在无候选时忙等 0.01s——源码 TODO 自己标注「可能阻塞 Alloc Loop 死锁」，是 LMCache PD reservation 要解决的问题在 Mooncake 侧的镜像（尚未修完）。

## 4. 三种哲学对照

| | LMCache PD reservation | SGLang no-over-allocation | Mooncake 四道闸门 |
|---|---|---|---|
| 主攻敌人 | **死锁** | **抖动** | **饿死/公平** |
| 决策时机 | 提前（首批全量预留） | 滞后（命中数确定后） | 持续（水位+频率滚动评估） |
| 分配单位 | 逻辑请求（all-or-nothing） | 确认命中的页（精确） | 对象级（频率驱动） |
| 资源不足时 | 整体等待 | 降级链（截断→revoke） | 拒绝提升/触发驱逐 |
| 类比 | 数据库 extent 预留 | 按需分页（demand paging） | QoS 水位治理 |

**不是谁更好，是各自命门不同**：LMCache 的 PD 场景里部分完成=白干（正确性），必须事务式；SGLang 的预取场景里多预留=纯浪费（数据可能根本没命中），必须反馈式；Mooncake 的共享池场景里没有单一「请求」可以预留（成千上万的 key 各自独立），只能服务端滚动治理。**判断题**：把 Mooncake 的事务式预留搬到 SGLang 预取上会怎样？——host 内存大量被「可能不命中」的预留占用，抖动回来；把 SGLang 的滞后分配搬到 LMCache PD 上会怎样？——死锁回来。**机制和场景是绑定的，抄方案不抄前提就是事故**。

## 5. 收束

三家的共性洞察：**admission control 的本质不是「限流」而是「保证逻辑单元的可完成性」**——LMCache 用预留保证请求可完成、SGLang 用精确分配保证热数据不被挤占、Mooncake 用闸门保证提升值得做。呼应[异步保存与块保护](/core/async-protection)：那篇讲「数据维度的保护」（不被释放），本篇讲「空间维度的保护」（不被挤占）——一个异步 KV 系统的稳定性 = 两者缺一不可。

> 🎯 **面试考察点**（66 家公司真题库）：「PD 分离调度队列怎么设计？」「预取会不会把缓存挤爆？」「高并发下系统为什么会挂死？」——追问链：死锁场景构造 → 三家方案 → 场景绑定判断。调度类题的深水区，答出「三个敌人三种哲学」即是区分度。相关真题见[面试题库](/interview/inference-answers)。
