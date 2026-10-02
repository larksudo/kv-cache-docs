---
prev: /core/transfer-paths
next: /core/storage-transforms

---

本篇回答一个底层问题：host 进程（CPU 侧代码）是如何「感知」并搬运 GPU 显存中的 KV Cache 的。五种地址辨析、pinned memory 的真实作用、zero-copy 的确切含义、RDMA 注册机制与生命周期协议，都在这一篇。

> 来源：V11 全文。文中 LMCache/Mooncake 源码引用基于 2026-08-28 快照。

# Host 如何感知 GPU 地址：Pinned Memory、Zero Copy 与 RDMA

本章回答一个更底层的问题：**host 真正“感知”到了什么？** 当代码写出：

```python
base_addr = cache.data_ptr()
kv_data_ptrs.append(base_addr)
kv_data_lens.append(cache.nbytes)
```

host 并没有拿到一段可以被 CPU 解引用的 GPU 内存。它拿到的是一个整数形式的 **GPU 虚拟地址**。这个地址之所以能用于传输，是因为后续 driver、CUDA Runtime、IOMMU、NIC、TransferEngine 或 NIXL 把它注册成可 DMA 访问的内存区域。


> **🎯 面试考察点**（66 家公司真题库 · 零拷贝与寻址）：
> - 「五种地址的区别」「pinned memory 解决什么」「zero-copy 到底拷不拷」；
> - 追问链：pinned vs pageable 的 DMA 语义 → 注册表如何跨节点生效 → 生命周期比地址计算难在哪。
> 接口级细节见[H2D/D2H 深潜](/core/h2d-d2h-deepdive)

## 1. 先分清五种“地址”

| 名称 | 谁能直接使用 | 例子 | 生命周期约束 |
|---|---|---|---|
| CPU virtual address | 当前 host 进程 | CPU `MemoryObj.data_ptr()` | 只要页表映射存在；普通 heap 内存可能被 OS 换出 |
| pinned host address | CUDA DMA、NIC、当前 host 进程 | `cudaHostAlloc/cudaHostRegister` 后的 host pointer | 页被锁定；注册期间不能 unmap/free |
| GPU device virtual address | GPU kernel、CUDA Runtime、GPU Direct 数据面 | `torch.Tensor.data_ptr()`，device 是 CUDA | tensor 存活且 stream/event 正确同步 |
| DMA/IOVA/bus address | NIC、DMA engine、IOMMU | 注册后由 driver/IOMMU 生成 | 与 registered memory region 绑定 |
| MR/descriptor/rkey | NIC、TransferEngine/NIXL | `register_memory()` 后的 descriptor、lkey/rkey | deregister 后失效 |

最常见的误解是把 `data_ptr()` 当成“host 可以读的地址”。实际上：

```text
host 进程看到的是地址值，
CUDA/driver/IOMMU/NIC 看到的是可翻译、可 DMA 的内存区域。
```

host 可以保存它、比较它、把它发给远端，但不能用它读取 GPU 数据。真正读数据的是 GPU kernel、copy engine、NIC 或远端 DMA。

## 2. Pinned Memory 到底解决了什么

普通 pageable host memory 有两个问题：

1. OS 可能换出或移动页；  
2. DMA engine 需要稳定、锁定的物理映射。  

Pinned memory 通过 `cudaHostAlloc()` 或 `cudaHostRegister()` 页锁定。LMCache 的相关路径：

| 组件 | 文件 |
|---|---|
| CPU pool 分配 | `LMCache/lmcache/v1/memory_management.py` |
| pinned allocator | `LMCache/lmcache/v1/memory_allocators/pin_memory_allocator.py` |
| NUMA/hugepage pinned 分配 | `memory_management.py::_allocate_cpu_memory()` |
| `cudaHostRegister` | `LMCache/lmcache/v1/platform/cuda/pin_memory.py` |

LMCache 不是每个 chunk 都临时 malloc。它通常先分配一个大 pinned pool，再由 `PagedTensorMemoryAllocator/TensorMemoryAllocator` 切出 `TensorMemoryObj`。每个对象仍带 shape/dtype/format，但底层缓冲区的地址在传输期间稳定。

Pinned memory 的收益：

1. `cudaMemcpyAsync` 可以真正异步；  
2. 避免 pageable memory 的隐式 staging copy；  
3. NIC/RDMA 可以注册这段内存；  
4. io_uring/GDS/持久化路径也能使用稳定 buffer；  
5. NUMA 感知分配可以减少跨 socket PCIe 访问。  

注意：**pinned 不等于 zero-copy**。Pinned 是“页锁定、可稳定 DMA”；zero-copy 通常指“避免 CPU 参与的数据复制或序列化”。

## 3. LMCache GPU ↔ CPU 的搬运

### 3.1 常见进程内路径

LMCache 的 vLLM connector 先注册引擎 KV cache：

```text
layer_name -> paged GPU tensor
paged tensor -> data_ptr / group pointers
```

例如 `VLLMPagedMemGPUConnectorV2/V3` 会保存各层或各 group 的 `data_ptr()`，并把这些指针上传成 GPU pointer tensor，供自定义 kernel 使用。`from_gpu()` 的典型流程是：

```text
vLLM paged KV
  -> GPU kernel/copy engine 汇聚到 GPU staging buffer
  -> cudaMemcpyAsync / tensor.copy_(non_blocking=True)
  -> pinned host MemoryObj
  -> L1 backend
```

`to_gpu()` 反向执行：

```text
CPU pinned MemoryObj
  -> GPU staging buffer
  -> paged GPU KV
```

源码中反复出现两类模式：

```python
tmp_gpu_buffer = self.gpu_buffer[:, :, start:end, :]
memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)
```

以及：

```python
ptrs = get_group_data_ptrs(...)
self.kv_cache_pointers_on_gpu.copy_(self.kv_cache_pointers)
```

前者表示把 GPU staging 结果 DMA 到 pinned CPU tensor；后者表示把 host 侧收集到的 GPU 地址交给 GPU kernel 使用。

### 3.2 为什么 GPU staging buffer 常常存在

理想情况是 paged KV 直接从 HBM 拷贝到 host。但 engine layout 可能是：

1. K/V 分离；  
2. 多层分散；  
3. HND/NHD 不同；  
4. MLA fused；  
5. cross-layer；  
6. TP shard 分布在不同 rank。  

GPU staging buffer 可以先把分散的 paged block 聚合成连续 chunk。这样 CPU 端 `MemoryObj` 是一块连续区域，后续序列化、RDMA、SSD offload 更简单。

代价是可能多一次 GPU 内 copy。LMCache 的 layerwise 路径则用 generator/stream 把“第 N 层搬出”和“第 N+1 层计算”重叠。

### 3.3 这里的 zero-copy 是什么意思

LMCache 中的 zero-copy 通常指以下几类：

| 层次 | 含义 | 是否等于没有数据拷贝 |
|---|---|---|
| serialization zero-copy | 不把 tensor 序列化成 Python bytes | 否，仍会 GPU→CPU DMA |
| CPU buffer zero-copy | 直接从 `MemoryObj.data_ptr()` 发送 | 否，数据会经 DMA |
| CUDA IPC zero-copy | server 直接读写 worker GPU 显存 | 否，设备/拷贝引擎仍搬数据，但避免 CPU 中转 |
| GPUDirect RDMA zero-copy | NIC 直接读写 GPU HBM | 数据仍会通过网络传输，但避免 CPU/host staging |

所以更准确的说法是：  
**pinned memory + descriptor 是 CPU 参与最小化；GPUDirect RDMA 才能减少 host staging；没有任何方案能让数据“不发生搬运”。**

## 4. Host 如何把 GPU 地址交给 NIC

host 不能把 `data_ptr()` 直接解释成物理地址。典型流程是：

```text
torch.Tensor.data_ptr()
  -> CUDA/驱动确认这是 GPU virtual address
  -> RDMA/GDR driver 将 GPU buffer 注册为 memory region
  -> IOMMU/GDR 建立设备可见映射
  -> NIC 获得MR/descriptor/key
  -> work request 使用 addr + length + key
```

在 NIXL/TransferEngine 这类抽象下，用户通常看不到裸 `lkey/rkey`，而是看到：

```text
reg_desc / xfer_desc / local_handle / remote_handle
```

LMCache 的 NIXL 路径就是这种模型：

| 步骤 | 源码位置 |
|---|---|
| 描述 L1 buffer | `lmcache/v1/distributed/internal_api.py::L1MemoryDesc` |
| 注册 CPU/GPU 内存 | `nixl_store_l2_adapter.py::init_mem_handlers()` |
| 构造 transfer descriptor | `get_xfer_descs()` |
| 生成 prepped handle | `make_prepped_xfer()` |
| 提交/轮询 | `transfer()` / `check_xfer_state()` |

NIXL store adapter 会按 page 把 buffer 注册：

```python
xfer_desc = [
    (base_addr, page_size, device_id)
    for base_addr in range(buffer_ptr, buffer_ptr + buffer_size, page_size)
]
if device.type == "cpu":
    mem_type = "DRAM"
elif device.type == "cuda":
    mem_type = "VRAM"
reg_descs = nixl_agent.register_memory(reg_list, mem_type=mem_type)
```

`DRAM` 与 `VRAM` 的区别就在这里。CPU pinned memory 注册后可被 NIC DMA；CUDA VRAM 注册后，若硬件/驱动支持，则可走 GPUDirect RDMA。

## 5. LMCache P2P：跨节点读 peer 的 pinned L1

LMCache P2P 的目标是共享 peer 的 L1 CPU memory。流程不是“远端进程读自己的内存再发过来”，而是：

```text
本地请求 key
  -> MQ lookup-and-lock
  -> peer 返回 TransferChannelAddress(offset,size)
  -> 本地准备 pinned MemoryObj
  -> submit_read(local_addresses, remote_addresses)
  -> NIXL/RDMA read
  -> query_read_status
  -> unlock
```

关键点：

1. peer 先锁对象，防止 RDMA 读期间被驱逐；  
2. 远端 L1 buffer 已经注册；  
3. 控制面只传 offset/size/descriptor，不传 KV 字节；  
4. 数据由 NIC 从 peer pinned memory DMA 到本地 pinned memory；  
5. 传输失败时退化为 miss，而不是返回不确定数据。  

相关源码：

```text
LMCache/lmcache/v1/distributed/l2_adapters/p2p_l2_adapter.py
LMCache/lmcache/v1/distributed/transfer_channel/impl/nixl_impl.py
```

P2P 这里之所以选择只读，是因为远端写会引入远程分配、对象存在性检查、失败回滚和双节点状态机。只读让失败边界更清晰：锁不到、读超时、bitmap miss。

## 6. LMCache PD：reservation + staging + RDMA

LMCache PD 分离有两个关注点：

1. 数据怎么搬；  
2. receiver staging buffer 怎么保证不会死锁。  

### 6.1 数据面

典型路径：

```text
Prefill GPU paged KV
  -> sender staging pinned buffer
  -> receiver 预分配/预留
  -> RDMA write/read
  -> receiver staging
  -> receiver GPU paged KV
```

sender 侧的 staging buffer 也是 pinned memory。GPU connector 先把 chunk 写入 sender staging；控制面通过 ZMQ 发送 `AllocRequest/AllocResponse/ProxyNotif/CancelNotif`；RDMA/NIXL 负责跨节点搬运。

### 6.2 admission control

如果只按 chunk 逐批分配，receiver 可能出现：

```text
请求 A 占了一半
请求 B 占了一半
A 等剩余空间
B 等剩余空间
双方死锁
```

LMCache 的 `ReservationManager` 在首个 batch 预留 `total_chunks`。之后：

1. receiver 保证后续 chunk 有空间；  
2. sender staging 用条件变量做物理背压；  
3. abort/failure 全量回滚；  
4. ProxyNotif 等所有 chunk 和 last batch 都完成才发送。  

这与 RDMA 生命周期紧密相关：只要 transfer 可能还在进行，源 buffer 和目标 buffer 都不能释放。

## 7. Mooncake direct PD：GPU `data_ptr()` 如何跨节点生效

以 Mooncake connector 为例：

```python
for layer_name, cache_or_caches in kv_caches.items():
    cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
    for cache in cache_list:
        base_addr = cache.data_ptr()
        kv_data_ptrs.append(base_addr)
        kv_data_lens.append(cache.nbytes)

self.kv_caches_base_addr = seen_base_addresses
engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
```

这段代码做了三件事：

1. 把每层 K/V 或 fused KV 的 GPU virtual address 收集成 region list；  
2. 把地址和长度交给 Mooncake `batch_register_memory()`；  
3. Mooncake TransferEngine/driver 将这些 GPU region 注册为 RDMA 可访问区域。  

注册完成后，`kv_caches_base_addr` 不再只是 Python 数字，而是代表一组已注册的 GPU region descriptor。远端不能凭裸地址访问它；它必须拿到 Mooncake 交换的 segment/remote descriptor，再通过 TransferEngine 发起操作。

### 7.1 控制面交换什么

D/P 双方先通过 ZMQ/msgpack 交换类似下面的 metadata：

```python
MooncakeAgentMetadata(
    remote_hostname=...,
    remote_port=...,
    request_ids=[...],
    kv_caches_base_addr=[...],
    block_ids=[...],
)
```

新版本还会带：

```python
block_lens
kv_block_lens
registered_layer_names
```

这些字段用于解决：

1. K/V 分离或 MLA fused；  
2. 不同层 region 大小不同；  
3. TP 不同导致 region 长度不同；  
4. PP 导致 positional matching 不安全；  
5. producer/consumer layer/group 对齐。  

### 7.2 数据面怎么计算

发送侧构造：

```text
src_ptr = local_base_addr[layer] + local_block_id * block_len
dst_ptr = remote_base_addr[layer] + remote_block_id * block_len
length  = contiguous_block_count * block_len
```

然后提交：

```python
engine.batch_transfer_sync_write(
    remote_session,
    src_ptrs,
    dst_ptrs,
    lengths,
)
```

这看起来像“远端 GPU 地址直接写入”，但实际依赖的是：

1. 两端都已经注册内存；  
2. Mooncake/驱动已经交换 segment/descriptor；  
3. NIC 能翻译本端和远端 registered region；  
4. 目标 tensor 在完成前不会被释放或重用。  

如果目标是远端 GPU 显存，则要求 GPUDirect RDMA 路径可用；如果不可用，系统必须退化为 CPU staging 或其他路径。

### 7.3 K/V 为什么可以被同等对待

在数据面中，K 和 V 都只是连续 region：

```text
layer 0 K: base_addr[0] + block_id * block_len
layer 0 V: base_addr[1] + block_id * block_len
layer 1 K: base_addr[2] + block_id * block_len
layer 1 V: base_addr[3] + block_id * block_len
```

传输层不需要知道 K/V。只要注册顺序和 block plan 一致，搬运就正确。K/V 语义由 vLLM/Mooncake connector 的 region 计划保存。

MLA 情况下，K/V 已经在引擎层融合为一个张量，Mooncake 只看到一个 region。FlashInfer/Pallas 也可能出现 fused layout。因此 `split_k_and_v` 不是 Mooncake 的数学判断，而是对引擎布局的适配。

## 8. Mooncake Store 与 pinned buffer

Mooncake Store 路径与 direct PD 不同。它通常把 KV 写入共享对象池：

```text
GPU KV
  -> connector 侧 buffer
  -> batch_put_from_multi_buffers()
  -> Mooncake Store object/slices
```

如果源是 CPU pinned buffer，connector 会先注册 CPU buffer：

```python
buffer_ptr = buffer.data_ptr()
store.register_buffer(buffer_ptr, buffer.numel())
```

之后 `batch_put_from()` / `batch_get_into()` 就能使用这个指针。此时 Mooncake Store 看到的是 registered buffer 中的连续字节，而不是 LMCache `MemoryObj` 的 shape。

在 LMCache 接 Mooncake 的实现里，存在两种模式：

| 模式 | put/get 形式 | 元数据 |
|---|---|---|
| zero-copy | `put_from()` / `batch_get_into()` | LMCache 本地保存 shape/dtype/format |
| metadata-embedded | `put_parts()` / `batch_get_buffer()` | `RemoteMetadata + kv_bytes` 一起存远端 |

前者传输更直接，但读取端必须能从本地元数据重建对象；后者远端自描述，但头部会占用额外字节。

## 9. GPU Direct RDMA 与 CPU staging 的区别

### 9.1 普通 LMCache CPU offload

```text
GPU HBM
  -> PCIe DMA
  -> host pinned DRAM
  -> optional NIC RDMA
  -> remote host DRAM
```

如果跨节点，数据至少经历两次主要 DMA：

1. GPU HBM -> local pinned DRAM；  
2. local pinned DRAM -> remote DRAM。  

这是很多系统的稳定路径，因为它对 NIC/GPU 兼容性要求低。

### 9.2 GPUDirect RDMA

```text
GPU HBM
  -> NIC DMA
  -> remote GPU HBM / remote DRAM
```

理想情况下，host CPU 只参与控制面，不搬运数据。Mooncake direct PD 想达到的就是这种效果。

前提条件：

1. NIC 与 GPU 在同一 PCIe switch/NUMA 域；  
2. GDR driver/IOMMU 配置正确；  
3. GPU buffer 注册成功；  
4. block 地址连续且对齐；  
5. CUDA stream/forward 完成后再传输；  
6. 远端目标 region 已注册且生命周期受保护。  

### 9.3 选择建议

| 场景 | 更合适 |
|---|---|
| 兼容性优先 | GPU -> pinned host -> RDMA |
| 小块分散 layout | 先 GPU 聚合，再一次性 DMA |
| 大块连续 KV、GDR 可用 | GPUDirect RDMA |
| 需要 crash isolation | 独立 CPU daemon + pinned pool |
| 需要低 CPU overhead | registered memory + descriptor |
| 需要跨引擎稳定 | 保留 MemoryObj/shape 元数据 |

## 10. 生命周期：地址有效不等于数据有效

RDMA 最危险的不是地址算错，而是地址仍在但数据已经变了。

必须保证：

1. GPU tensor 不被释放；  
2. paged block 不被 allocator 复用；  
3. pinned host buffer 不被 unpin/free；  
4. registered memory 不在 transfer 完成前 deregister；  
5. CUDA stream/event 已同步；  
6. 多 rank 都完成后再通知对端；  
7. abort 时释放目标 reservation。  

Mooncake Store 的 `store_job_id` 和 GPU block reference 就是为了解决这个问题：request id 可能复用，store job 不应复用；所有 rank 完成前，scheduler 不能释放被 DMA 读取的 block。

LMCache 的 P2P 使用 peer lookup-and-lock；PD 使用 reservation/staging flow control。两者的本质相同：**在异步 DMA 完成前，给数据区一个显式生命周期。**

## 11. 一个完整心智模型

可以把一次跨节点 KV 传输拆成四张表：

### 11.1 逻辑表

```text
request prefix
  -> chunk/block hash
  -> KV tensor layout
  -> TP/PP/CP shard
```

### 11.2 本地地址表

```text
region id -> base address
block id  -> offset
length    -> nbytes
device    -> CUDA / CPU
state     -> ready / in-flight / completed
```

### 11.3 注册表

```text
ptr,size,device
  -> MR/descriptor
  -> lkey/rkey or NIXL descriptor
  -> peer-visible remote handle
```

### 11.4 传输表

```text
transfer id
  -> src descriptor + offset
  -> dst descriptor + offset
  -> length
  -> completion state
```

host 主要维护前三张表；NIC 和 driver 消费第四张表。

## 12. 排障清单

| 症状 | 检查 |
|---|---|
| `batch_register_memory` 失败 | GPU Direct/RDMA 驱动、IOMMU、NIC-GPU 亲和、地址对齐 |
| RDMA 建链失败 | metadata server、端口、防火墙、GID、RoCE PFC/ECN |
| 带宽只有单 NIC | topology matrix、multi-rail、slice spraying、NUMA |
| 数据错乱 | region 顺序、K/V split、layer name、block id、TP/PP 对齐 |
| 偶发脏数据 | tensor 生命周期、stream/event、block reuse、registered memory 提前 deregister |
| pinned copy 不异步 | 是否真正 pinned、CUDA stream、pageable staging、NUMA |
| P2P 偶发 miss | lookup timeout、peer lock、对象驱逐、transfer deadline |
| PD 卡死 | `total_chunks` reservation、staging buffer、abort 回滚 |

## 13. 最小实验

### 实验 1：验证 pinned 是否生效

记录：

```text
host pointer
page size
allocation backend
cudaPointerGetAttributes result
H2D/D2H bandwidth
CPU utilization
```

对比 pageable 与 pinned。

### 实验 2：验证 GPU region registration

记录：

```text
每个 layer 的 base_addr
block_len
nbytes
registration return code
NIC/GPU BDF
IOMMU 状态
```

确认 K/V fused 与分离两种 layout。

### 实验 3：验证跨节点 metadata

抓取 ZMQ metadata，确认：

```text
remote hostname/port
base addr list
block id list
layer name list
block_len/kv_block_len
transfer id
```

再对照 RDMA/TransferEngine 日志。

### 实验 4：验证生命周期

在 transfer 完成前尝试：

1. 释放源 tensor；  
2. 复用目标 GPU block；  
3. unpin host buffer；  
4. deregister MR。  

正确实现应当阻止或延迟这些操作；否则会出现脏数据。

## 14. 结论

1. **host 感知的是地址和控制信息，不是 GPU 数据本身。**  
   `cache.data_ptr()` 只是 GPU virtual address。它能用于 RDMA，是因为 driver/TransferEngine/NIXL 把它注册成 NIC 可翻译的 memory region。

2. **pinned memory 的核心是稳定 DMA。**  
   它让 `cudaMemcpyAsync` 可靠异步，让 L1 pool 可以被 NIC/GDS/持久化路径安全访问，并避免 pageable memory 的额外 copy。

3. **zero-copy 是消除 CPU 参与和序列化，不是消除数据搬运。**  
   LMCache 常见路径仍有 GPU DMA 和 NIC DMA；GPUDirect RDMA 才能减少 host staging。

4. **跨节点依赖注册表，而不是裸地址。**  
   Mooncake/NIXL 都会交换 descriptor、segment 或 remote handle。远端不能只凭 `base_addr` 访问本端。

5. **生命周期协议比地址计算更难。**  
   block pin、store job、reservation、abort rollback、rank completion 才是生产系统中最容易出错的部分。

> **版本勘误（源码复核）**：`split_k_and_v = not (use_mla or _use_pallas_v1 or _use_flashinfer)` 等 connector 代码位于 Mooncake 自维护的 wheel 包 `Mooncake/mooncake-wheel/mooncake/mooncake_connector_v1.py:702`；当前 vllm 仓库内的 `vllm/.../mooncake/mooncake_connector.py` 已演进为按 storage 注册地址 + TransferRegion 机制，不再含 `split_k_and_v`。「80 层 × K/V = 160 个地址」是 80 层模型的示例值而非代码常量（代码按 base_addr 去重）。引用相关机制时请注明 wheel 版路径与版本差异。
