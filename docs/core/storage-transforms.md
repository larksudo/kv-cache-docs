---
prev: /core/zero-copy
next: /core/prefix-hit

---

本篇单独回答一个问题：同一条 KV Cache 在 LMCache 与 Mooncake 的不同存储引擎/介质之间，到底变成了什么——从 GPU paged block 到 CPU MemoryObj、RDMA descriptor、远端对象、Slice/Replica，再到 SSD Bucket 记录。

> 来源：V10 全文，以微信文章《一条数据的六次变身》为引，用本地源码复核其结构判断。

# KV Cache 在不同存储引擎中的变形：LMCache 与 Mooncake

本章单独回答一个问题：同一条 KV Cache 在 LMCache 与 Mooncake 的不同存储引擎/介质之间，到底变成了什么。参考资料是文章《KV Cache 在 Mooncake 系统中到底长什么样？从 Attention 公式到 SSD 磁盘上的字节，一条数据的六次变身》，本地快照为 `references/raw/wechat_1.html` / `.txt`；本章同时用本地源码复核其结构判断。

结论先说：**KV Cache 的数学值通常没有变，变化的是“可寻址单元、元数据、生命周期和一致性边界”。** LMCache 倾向于保留 `chunk + MemoryObj + shape/dtype/format` 的语义；Mooncake 则更早把数据抽象成地址区间、Slice、对象、Replica 和 Bucket 记录，让存储/传输层只关心连续字节。

---


> **🎯 面试考察点**（66 家公司真题库 · 存储变形）：
> - 「同一条 KV 在 GPU/CPU/SSD/远端各是什么形态？」（Mooncake 面试常问）；
> - 追问链：chunk vs block 语义 → 六次变身 → 谁为 layout 买单（LMCache vs Mooncake 的 gather vs 地址计算对比）。
> 对比精答见[H2D/D2H 深潜 §5](/core/h2d-d2h-deepdive)

## 1. 逻辑形态：K/V 矩阵不是文件格式

从 attention 计算看，一个 token、一层、一个 KV head 的逻辑元素是：

```text
K[token, layer, kv_head, head_dim]
V[token, layer, kv_head, head_dim]
```

聚合后常见为：

```text
K: [seq_len, num_kv_heads, head_dim]
V: [seq_len, num_kv_heads, head_dim]
```

但这只是数学视角。进入引擎后，它会立即被 paged block、TP shard、MLA 融合、attention backend、cache chunk 边界重新组织。因此把 KV Cache 理解为“一个数组”不够，必须同时追踪：

1. **逻辑键**：模型、tokenizer/revision、量化、adapter、prefix/block hash、layout；  
2. **物理布局**：NHD/HND、layer-first/page-first、K/V 分离或融合；  
3. **可寻址单元**：tensor、paged block、chunk、object、Slice、file record；  
4. **生命周期**：write/read lock、lease、replica、eviction、SSD offload；  
5. **一致性边界**：partial 写是否可见、多 rank 是否齐备、对象何时 COMPLETE。  

LMCache 与 Mooncake 的差异，本质上是它们在上述五点上的切分点不同。

---

## 2. LMCache：从 engine tensor 到可搬运 MemoryObj

### 2.1 CacheEngineKey：把前缀变成 chunk identity

LMCache 的核心身份不是 layer K/V 张量，而是 `CacheEngineKey`。源码 `LMCache/lmcache/utils.py` 中键的字符串形态大致为：

```text
model_name@world_size@worker_id@chunk_hash@dtype[@tag%value...]
```

也就是说，LMCache 在引擎 paged block 之上先建立“chunk”语义：若干 token 组成一个可缓存、可命中、可替换的逻辑单元。`LayerCacheEngineKey` 还可以继续带 `layer_id`，适配 layerwise 路径。

这一步的意义是：**存储后端不再理解完整请求，只需要理解 chunk key。**

### 2.2 MemoryObj：带形状的物理值

LMCache 的 L1 CPU 缓存不是裸 `bytes`，而是 `MemoryObj`。`LMCache/lmcache/v1/memory_management.py` 中：

- `MemoryObjMetadata` 保存 logical shape、dtype、地址、physical size、format、pin count 等信息；  
- `TensorMemoryObj` 包装 raw tensor；  
- `MemoryFormat` 支持 `KV_2LTD`、`KV_T2D`、`KV_2TD`、`KV_MLA_FMT`、`BINARY` 等多种布局。  

这使 LMCache 能在 GPU connector、CPU pool、local disk、GDS、NIXL、remote backend 之间传递“带语义的字节区”。代价是：每个后端必须理解 `MemoryObj`，或至少能恢复/保存其元数据。

### 2.3 GPU 层：paged block -> chunk tensor

vLLM 侧的 KV 仍然是 paged GPU tensor。一个 paged block 的物理地址通常满足：

```text
addr = tensor_base + block_id * block_len
```

但 LMCache 不把所有 engine layout 硬编码在存储层。它的 GPU connector 层先做 layout discovery：

- vLLM FlashAttention / FlashInfer；  
- HND/NHD；  
- MLA；  
- cross-layer；  
- fused/unified KV。  

相关源码在 `LMCache/lmcache/v1/gpu_connector/gpu_connectors.py` 和 `docs/design/v1/gpu_connector/layout-invariant.md`。也就是说，GPU paged block 先被规范化为 LMCache 可搬运的 page buffer/chunk，再交给存储路径。

### 2.4 CPU/L1：chunk key -> pinned MemoryObj

`LocalCPUBackend` 把 chunk key 映射到内存对象：

```text
CacheEngineKey -> MemoryObj
```

源码路径：`LMCache/lmcache/v1/storage_backend/local_cpu_backend.py`。

这一步发生三个变化：

1. **粒度变化**：engine paged block 变成 LMCache chunk；  
2. **介质变化**：GPU HBM 变成 CPU/pinned DRAM；  
3. **语义保留**：shape、dtype、format 仍在 `MemoryObjMetadata` 中。  

L1 中的对象是可 pin 的。读取路径可以拿到 `MemoryObj.tensor` 并 `batched_to_gpu()` 写回引擎。

### 2.5 Local Disk：MemoryObj -> file + DiskCacheMetadata

`LocalDiskBackend` 的变化更明显：

```text
MemoryObj
  -> bytes / file
  -> DiskCacheMetadata(path, size, shape, dtype, fmt, cached_positions, pin_count)
```

源码路径：`LMCache/lmcache/v1/storage_backend/local_disk_backend.py`。

读取时不是先读完整文件再猜格式，而是先查 metadata，再预分配 CPU `MemoryObj`，最后把文件内容读入该 buffer。因此磁盘上的对象虽然通常是字节序列，但 LMCache 会重新还原成带 shape 的 chunk。

### 2.6 GDS：GPU/MemoryObj -> NVMe 对齐记录

GDS backend 使用 `CuFile` / GPUDirect Storage，让 GPU 指针直接与 NVMe 交互，减少 CPU 中转。

关键源码：`LMCache/lmcache/v1/storage_backend/gds_backend.py`。

它的存储形态与普通 local disk 不同：

1. 数据文件与 `.metadata` 分离；  
2. metadata 固定预留 4KB；  
3. GPU buffer 需要对齐/注册；  
4. 写入路径更接近“GPU 地址 -> 文件区间”，而不是“CPU bytes -> 文件”。  

这里 KV Cache 变成的是 **NVMe 文件中的对齐原始张量字节 + 独立 metadata 文件**。

### 2.7 NIXL Store：MemoryObj -> registered pages / storage object

NIXL 路径把 L1 CPU/GPU buffer 注册成 transport/storage 可寻址区域。`nixl_storage_backend.py` 和 `distributed/l2_adapters/nixl_store_l2_adapter.py` 中，数据进一步变成：

```text
buffer ptr/size
  -> page indices
  -> storage file/object index
  -> DMA descriptor
```

此时 LMCache 的 chunk 语义仍在上层，但 NIXL 数据面看到的是 descriptor/page，而不是 K/V。

### 2.8 Redis/S3：MemoryObj -> kv_bytes + metadata

在 Redis/Valkey 后端中，LMCache 常见地把一个 chunk 拆成两类条目：

```text
key:kv_bytes   -> 原始 KV 字节
key:metadata   -> shape/dtype/format 等元数据
```

在 S3 后端中则是对象 PUT/GET。LMCache 会用 `memoryview`、`ctypes` 等方式避免额外复制，但语义上仍然是“带元数据的对象”。

### 2.9 Mooncake backend：MemoryObj -> Mooncake object

LMCache 也可以把 Mooncake Store 当作 remote backend：

源码：`LMCache/lmcache/v1/storage_backend/connector/mooncakestore_connector.py`。

这里出现两种模式：

1. **zero-copy 模式**：  
   `MemoryObj.data_ptr()` 和 `get_size()` 传给 `batch_put_from()` / `batch_get_into()`；Mooncake 不关心 shape。  

2. **metadata-embedded 模式**：  
   LMCache 把 `RemoteMetadata + kv_bytes` 组合成 `put_parts()` 的输入；读取时先解析 `RemoteMetadata`，再分配正确 shape/dtype 的 `MemoryObj`。  

这说明即使后端是 Mooncake，LMCache 仍然尽量把“chunk shape”恢复回上层。Mooncake 本身则只负责对象和 Slice。

---

## 3. Mooncake：从 GPU 地址到对象、Slice、Replica、Bucket

参考文章把 Mooncake 的数据旅程归纳为六次变形：

1. Attention 公式中的 K/V 矩阵；  
2. GPU 显存中的 paged block；  
3. RDMA 线路上的原始字节流；  
4. DRAM 中的 Slice；  
5. EP 分发中的 FP8 包；  
6. SSD 磁盘上的桶文件记录。  

需要补充一个边界：**EP FP8 包并不是严格意义上的持久 KV Cache 变形，而是 MoE dispatch 的 token/hidden-state 载荷**。它会与 KV cache 管理并存，但不应与 PutStart/PutEnd 管理的 KVCACHE 对象混为一谈。

### 3.1 direct PD：GPU paged block -> TransferRequest

Mooncake direct PD connector 的关键源码是：

```text
vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
```

vLLM worker 先注册 KV tensor：

```text
layer_name -> paged GPU tensor -> base_addr + nbytes
```

传输时，Mooncake 只需要：

```text
src_ptr = local_base + block_id * block_len
dst_ptr = remote_base + remote_block_id * block_len
length  = block_len * contiguous_block_count
```

于是 K/V 的身份被抹平成 `(source, target, length)`。这不是信息丢失，而是刻意下沉：GPU/RDMA 数据面不需要知道“这是 layer 3 的 K”，只需要知道搬哪些连续字节。

Mooncake 的优势在这里很明显：如果 block 布局可对齐，RDMA 路径可以避免中间拷贝和序列化。但代价是，producer/consumer 的 KV layout、TP/PP region、block size 必须严格匹配。

### 3.2 Store：chunk/block hash -> Mooncake object key

Mooncake Store Connector 不做 direct GPU-to-GPU，而是写入共享对象池。vLLM worker 源码：

```text
vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
```

它把 vLLM block hash 转成 Mooncake key，并可选生成 group id：

```text
vLLM block hash
  -> Mooncake object key
  -> optional group id
  -> batch_put_from_multi_buffers()
```

一个逻辑 prefix chunk 可以对应多个 Mooncake 对象，例如 K/V、TP shard、MLA component 或 hybrid attention sidecar。group id 用来描述“这些对象属于同一生命周期单元”。

这里 Mooncake 的抽象已经从“KV tensor”变成 **object + group**。

### 3.3 Master：object -> replica -> segment/slice

Mooncake Store 的写协议是：

```text
PutStart：Master 分配 replica/slice 空间
Client：通过 TransferEngine 写数据
PutEnd：Master 将 replica 标记 COMPLETE，对象可读
```

在 Master/存储层，KV Cache 变成：

```text
object key
  -> replica list
  -> segment buffer
  -> Slice(ptr, size)
```

当前源码 `Mooncake/mooncake-store/include/types.h` 中，`Slice` 只有两个字段：

```cpp
struct Slice {
    void* ptr{nullptr};
    size_t size{0};
};
```

`ObjectDataType::KVCACHE` 只是一个类型标签，用于未来做类型感知驱逐/复制策略；数据本身仍然是连续字节。文章说 Slice 约 16MB，当前源码更精确地写成：

```cpp
kMaxSliceSize = facebook::cachelib::Slab::kSize - 16;
```

也就是与 CacheLib slab 上限相关，而不是固定魔法数。

这一步是 Mooncake 与 LMCache 的核心差异：

- LMCache：后端通常拿到“一个带 shape 的 chunk”；  
- Mooncake Store：存储层拿到“object 的若干 Slice”，slice 可分布在不同 segment/node。  

这使 Mooncake 能把大对象并行写入/读取多个 DRAM 段，并配合多 NIC/RDMA 聚合带宽。

### 3.4 SSD：Slice -> Bucket file record

DRAM 空间不足时，Mooncake 可将对象 offload 到本地 SSD。SSD 层不是每个 KV block 一个小文件，而是把多个对象合并到 bucket file。

当前源码 `Mooncake/mooncake-store/include/storage_backend.h` 中：

```cpp
struct BucketObjectMetadata {
    int64_t offset;
    int64_t key_size;
    int64_t data_size;
};

struct BucketMetadata {
    int64_t meta_size;
    int64_t data_size;
    std::vector<std::string> keys;
    std::vector<BucketObjectMetadata> metadatas;
};
```

因此一条 KV Cache 在 SSD 上变成：

```text
bucket file
  -> data region
  -> key + offset + size
```

读取路径通常是 metadata 定位、O_DIRECT/对齐 staging、再回到 DRAM/GPU。此时“layer/block/KV”概念已经完全消失，只有 object key 和文件区间。

### 3.5 Replica：object 的持久/可用性形态

`Mooncake/mooncake-store/include/replica.h` 显示对象副本类型包括：

```text
MEMORY
DISK
LOCAL_DISK
NOF_SSD
DFS
```

副本状态包括 `PROCESSING`、`COMPLETE`、`REMOVED`、`FAILED`。这使 Mooncake 不只是把 KV Cache 写到另一个介质，而是维护“对象何时可读、在哪可读、是否有副本、何时被驱逐”的生命周期。

LMCache 的普通 remote backend 通常没有这么强的中心化对象状态机。LMCache 更依赖本地 store controller、backend 的 contains/put/get、L2 lock 和上层策略。

---

## 4. 逐介质对照表

| 阶段 | LMCache | Mooncake |
|---|---|---|
| attention 语义 | K/V tensor + engine layout | K/V tensor + engine layout |
| GPU paged block | engine tensor，由 GPU connector 规范化 | direct PD 直接计算 block 地址；Store connector 读 GPU buffer |
| 缓存单元 | chunk + `CacheEngineKey` + `MemoryObj` | vLLM block hash -> Mooncake object key/group |
| CPU/DRAM | pinned `MemoryObj`，保留 shape/dtype/format | segment 中的 `Slice(ptr,size)`，不保留 KV 语义 |
| 网络 | GPU connector/CUDA stream、NIXL descriptor、RDMA write/read、ZMQ 控制面 | TransferEngine `TransferRequest`、RDMA/NVLink/HCCS、multi-NIC slice |
| 远端对象 | Redis/S3/Mooncake 对象，常带 LMCache metadata | object + replica + group + optional SSD offload |
| SSD | local disk 文件、GDS 对齐数据 + metadata | bucket file + `BucketObjectMetadata` |
| 驱逐 | per-backend policy、L1/L2 controller、pin | master LRU/watermark、lease、soft/hard pin |
| 可见性 | 通常 put/get 后可读；后端语义不一 | PutStart/PutEnd，`PROCESSING -> COMPLETE` |
| 多副本 | 取决于后端，通常非核心 | best-effort replica，slice 分散放置 |
| 恢复 | local disk/GDS/NIXL dynamic 可恢复；远端取决于后端 | snapshot、OpLog、standby、SSD recovery |

---

## 5. 同一条数据在不同介质中的“名字”

以一个 LMCache chunk 为例：

```text
逻辑前缀
  -> CacheEngineKey(model@world_size@worker_id@chunk_hash@dtype)
  -> MemoryObj(shape,dtype,format,ptr,size)
  -> CPU pinned tensor
  -> disk file + DiskCacheMetadata
  -> NIXL page descriptors
  -> Redis/S3 kv_bytes + metadata
  -> Mooncake object key + buffer ptr/size
```

以一个 Mooncake Store 对象为例：

```text
vLLM block hash
  -> Mooncake object key
  -> PutStart 分配 replica/slice
  -> Slice(ptr,size) 写入 segment
  -> PutEnd 标记 COMPLETE
  -> replica type: MEMORY / DISK / LOCAL_DISK / NOF_SSD / DFS
  -> SSD bucket key + offset + size
```

可以看出：

- **LMCache 的“名字”更接近 application cache key**；  
- **Mooncake 的“名字”在存储层更接近 object address**。  

前者便于跨后端还原 KV shape，后者便于高带宽搬运和集群级对象治理。

---

## 6. 关键设计权衡

### 6.1 LMCache：语义保留优先

LMCache 保留 shape/dtype/format 的好处：

1. vLLM、SGLang、TRT-LLM 等引擎 layout 不同，仍然能还原；  
2. Redis、S3、local disk、Mooncake 等后端可以复用同一 cache key 抽象；  
3. lazy offload、layerwise、partial chunk、cached positions 等上层语义容易表达；  
4. 跨引擎和多云后端生态更容易扩展。  

代价：

1. `MemoryObj`/metadata 成为系统契约；  
2. 某些后端需要额外 metadata 条目或内嵌 header；  
3. 后端能力差异大，性能与一致性不完全等价；  
4. 想吃到 GPU Direct/RDMA 大带宽时，必须再进入 NIXL/GDS/Mooncake 这类 descriptor 路径。  

### 6.2 Mooncake：物理寻址优先

Mooncake 尽早抹平 K/V、layer、chunk 的语义，好处：

1. RDMA/TransferEngine 只搬运连续字节，路径短；  
2. Slice 可分布到多个 segment，配合多 NIC 聚合带宽；  
3. Store 可以统一管理 replica、lease、pin、eviction、SSD offload；  
4. 对象池可以承载 KV cache、tensor、weight、hidden state 等多种张量资产。  

代价：

1. producer/consumer layout 必须可精确映射；  
2. TP/PP、MLA、hybrid attention 会在 connector 层带来复杂校验；  
3. 上层必须保证异步 DMA 期间 GPU block 不被释放；  
4. 集中 Master/HA、对象生命周期和一致性协议带来更高工程复杂度。  

---

## 7. 写路径对比

### 7.1 LMCache 典型写路径

```text
vLLM paged GPU KV
  -> GPU connector normalize
  -> GPU staging / MemoryObj
  -> L1 CPU MemoryObj
  -> StoreController
  -> L2 backend:
       local disk / GDS / NIXL / Redis / S3 / Mooncake
```

如果后端是 Mooncake：

```text
L1 MemoryObj
  -> data_ptr,size 或 metadata+kv_bytes
  -> MooncakeDistributedStore.put_from / put_parts
  -> Mooncake Store object/slices
```

### 7.2 Mooncake direct PD 写路径

```text
Prefill vLLM paged GPU KV
  -> registered region / block ids
  -> TransferRequest(src,dst,len)
  -> RDMA / NVLink / HCCS
  -> Decode vLLM paged GPU KV
```

这条路径通常不经过 Mooncake Store object，也不产生 SSD bucket。

### 7.3 Mooncake Store 写路径

```text
Prefill/Decode GPU KV
  -> block hash -> Mooncake key
  -> batch_put_from_multi_buffers()
  -> PutStart
  -> TransferEngine 写 replica/slices
  -> PutEnd
  -> object COMPLETE
  -> optional SSD offload
```

---

## 8. 读路径对比

### 8.1 LMCache 读路径

```text
request prefix
  -> chunk keys
  -> L1 hit / L2 lookup
  -> backend get
  -> MemoryObj
  -> GPU connector
  -> vLLM paged GPU KV
```

如果来自 Mooncake：

```text
Mooncake object
  -> batch_get_into / batch_get_buffer
  -> LMCache MemoryObj
  -> GPU connector
  -> vLLM GPU KV
```

### 8.2 Mooncake direct PD 读路径

Decode 不是从对象池 get，而是把本地 paged block 地址交给 P 端，由 P 端发起 RDMA write，或 D 端按协议 pull。

```text
Decode local block ids
  -> transfer region
  -> TransferEngine
  -> local paged GPU KV
```

### 8.3 Mooncake Store 读路径

```text
prefix hash
  -> object key / replica list
  -> choose memory/disk/remote replica
  -> TransferEngine / zero-copy buffer
  -> vLLM paged GPU KV
```

---

## 9. 元数据在哪里

| 系统 | 元数据位置 | 典型字段 |
|---|---|---|
| LMCache CPU | `MemoryObjMetadata` | shape、dtype、ptr、phy size、format、pin count |
| LMCache local disk | `DiskCacheMetadata` | path、size、shape、dtype、fmt、cached positions、pin count |
| LMCache GDS | `.metadata` 文件 | tensor shape、dtype、nbytes、format、extra metadata |
| LMCache Redis/S3 | 独立 metadata 条目或对象头 | length、shape、dtype、format |
| LMCache Mooncake | zero-copy 依赖本地 metadata；兼容模式内嵌 `RemoteMetadata` | length、shapes、dtypes、format |
| Mooncake direct PD | connector metadata | hostname/port、base addr、block ids、TP/PP region |
| Mooncake Store | Master + Replica/Slice | object key、replica status、segment、offset、size、lease/pin |
| Mooncake SSD | BucketMetadata | key、offset、key_size、data_size |

这张表解释了为什么两者排障方式不同：

- LMCache 先查“chunk key 是否存在、MemoryObj 是否可还原”；  
- Mooncake 先查“object 是否 COMPLETE、replica 是否可用、Slice/Bucket offset 是否有效”。

---

## 10. 一致性与生命周期

### 10.1 LMCache

LMCache 的一致性主要来自：

1. chunk key 不可变；  
2. L1/L2 store/prefetch controller 的 read/write lock；  
3. P2P lookup-and-lock；  
4. backend 自身的 put/get 语义；  
5. MP daemon 的独立生命周期。  

但它不是所有后端都有统一事务。例如 local disk、Redis、S3、Mooncake 的失败回滚能力不同。

### 10.2 Mooncake

Mooncake Store 有更明确的对象生命周期：

```text
PutStart
  -> PROCESSING
  -> PutEnd
  -> COMPLETE
  -> lease / read
  -> eviction / remove
```

这带来：

1. partial object 不可见；  
2. lease 可保护读期间不被驱逐；  
3. soft/hard pin 可保护热点或权重；  
4. zombie PutStart 可被超时清理；  
5. snapshot/OpLog 支撑 Master 恢复。  

这也是把 KV cache 当数据库看时，Mooncake 更像分布式缓存存储引擎的原因。

---

## 11. 常见误读

### 误读一：RDMA 后就没有 layout

RDMA 上确实没有 K/V 字段，只有连续字节；但上层必须先解决 layout、block id、TP/PP region 对齐。语义没有消失，只是被移到 connector。

### 误读二：Store 里还能按 layer 读

Mooncake Store 的对象可以按 key 读，但存储层通常不提供“读 layer 3 的 K head 5”的语义。要做到细粒度读，需要上层把对象/键划分得更细，或在客户端做剪裁。

### 误读三：EP FP8 包是 KV Cache offload

EP FP8 包主要是 MoE dispatch 的 hidden/token 载荷，用于专家并行通信。它可能与 KV cache 系统共用底层网络，但生命周期和一致性语义不同。

### 误读四：LMCache 和 Mooncake 只是两个后端

LMCache 可以把 Mooncake 作为后端；Mooncake direct PD 与 Mooncake Store 也不是同一条路径。更准确的图是：

```text
LMCache = engine-aware cache management layer
Mooncake TE = transport/data plane
Mooncake Store = distributed object/cache store
```

---

## 12. 排障锚点

| 症状 | 应优先检查 |
|---|---|
| PD direct 传输慢 | GPU block 地址计算、连续性、TP/PP region、NIC/NUMA、是否合并 block |
| Mooncake Store 吞吐低 | Slice 大小、segment 分布、multi-NIC 利用率、batch_put_from_multi_buffers 参数 |
| LMCache L2 miss 但源站有数据 | CacheEngineKey、cache salt、world size/worker id、model revision、tags |
| 远端读回来 shape 错 | metadata 是否保存、`MemoryObjMetadata` 是否正确恢复 |
| GDS 读失败 | 4KB 对齐、GPU buffer 注册、文件系统、cuFile 配置 |
| Mooncake SSD 读慢 | BucketMetadata、O_DIRECT 对齐、staging buffer、SSD 队列深度 |
| 异步 store 读到旧块 | GPU block pin、store job、CoW/partial tail、rank 完成计数 |

---

## 13. 最小测试矩阵

| 测试 | 目标 |
|---|---|
| GPU -> LMCache CPU -> GPU | 验证 `MemoryObj` shape/dtype/format roundtrip |
| CPU -> local disk -> CPU | 验证 `DiskCacheMetadata` 恢复 |
| CPU/GPU -> GDS -> GPU | 验证对齐、metadata、零拷贝路径 |
| CPU -> NIXL Store -> CPU | 验证 page index/descriptor roundtrip |
| CPU -> Redis/S3 -> CPU | 验证 `kv_bytes + metadata` |
| CPU -> Mooncake -> CPU | 验证 zero-copy 与 metadata-embedded 两种模式 |
| GPU -> Mooncake direct PD -> GPU | 验证 block address、TP/PP region、异步完成 |
| Mooncake Store -> SSD -> Store | 验证 BucketMetadata、offload/restore |

每个测试都应记录 key、shape、dtype、format、字节数、copy 次数、成功/失败语义。

---

## 14. 容量不变，为什么 Slice 和 DRAM 分布还会影响吞吐？

一个常见误解是：既然 KV Cache 从 GPU block 变成 RDMA bytes、DRAM Slice、SSD Bucket 时数据量基本不变，那性能就应该只由容量决定。这个推断不成立。

存储系统的时间不是：

```text
time = bytes / 某个固定带宽
```

而是：

```text
time ≈ metadata/setup
     + max(每条并行路径的 transfer time)
     + tail/重试
     + CPU/锁/完成事件开销
```

同样的 160MB KV bytes，可以有多种读法：

| 形态 | 效果 |
|---|---|
| 10 × 16MB Slice，分布在 10 个健康 segment/NIC | 可并行聚合，接近 10 条路径带宽 |
| 10 × 16MB Slice，全部落在同一 segment/NIC | 容量不变，但带宽退化为单路径 |
| 1600 × 100KB Slice | 并发描述符、metadata、CQ、锁和调度开销显著上升 |
| 9 个 Slice 本地快、1 个 Slice 跨 NUMA/跨节点慢 | 整体 completion 被最慢 Slice 拖住 |
| Slice 与 NIC/NUMA 亲和错配 | RDMA 可能跨 UPI/PCIe root，实际带宽下降 |

Mooncake Store 文档明确支持大对象 striping/parallel I/O，并把对象 Slice 放置到不同 segment；Transfer Engine 又会按 CPU/GPU/NIC topology 选择路径，长请求会继续切成 slice，以利用多 RDMA NIC。因此，“容量不变”只说明逻辑 KV bytes 没少，不代表有效并行带宽、NUMA locality、metadata overhead 和 completion tail 相同。

### 14.1 Slice 大小为什么重要

Slice 太大：

1. 并行度不足，容易被单 NIC/单 segment 限制；  
2. 分配失败概率更高，尤其高水位/碎片化时；  
3. 失败重传代价大。  

Slice 太小：

1. descriptor、CQE、锁、completion、错误码数量增加；  
2. Master/client 侧 metadata 和 bookkeeping 变多；  
3. 小于 RDMA 有效负载/对齐收益时，协议开销占比高。  

因此吞吐良好的状态通常是：Slice 足够大以摊薄开销，又足够多/均匀以利用多路径；每个 Slice 与可用 NIC/NUMA 拓扑有较好的亲和。

### 14.2 DRAM 段分布为什么重要

这里说的不是“DRAM 容量够不够”，而是“每个 Slice 的注册内存位于哪个 segment/node/NUMA/NIC 域”。

好的分布：

```text
Slice0 -> node A / NIC0
Slice1 -> node B / NIC1
Slice2 -> node C / NIC2
...
```

差的分布：

```text
Slice0..N -> node A / NIC0
```

或者：

```text
多数 Slice 在远 NUMA
少数 Slice 在拥塞链路
```

前者是 aggregate bandwidth 问题，后者是 tail latency 问题。两者都会让“容量没变”的读/写变慢。

### 14.3 当前 Mooncake Store 实现中的一个边界

参考文章用“一个 160MB 请求对象切成 10 个 Slice”说明原理。当前 vLLM MooncakeStoreConnector 的主流写路径更常见的是：每个 block/chunk hash 对应一个 Mooncake object，并通过 `batch_put_from_multi_buffers()` 批量搬运。也就是说，实践中未必总有一个 160MB 的 request-level object。

但这不削弱结论：

1. 单个对象足够大时，Slice 大小和 segment/NIC 分布直接决定 striping 效率；  
2. 大量 block/chunk 小对象时，batch 大小、对象数量、Master RPC、segment 选择和 batch buffer 连续性同样决定吞吐；  
3. 两者本质都是“容量相同，但有效并行带宽和协议开销不同”。  

### 14.4 与 LMCache 的性能比较

不能脱离场景说“LMCache 更快”或“Mooncake 更快”。

| 场景 | 通常更合适 | 原因 |
|---|---|---|
| 单机 GPU -> local pinned CPU -> GPU | LMCache LocalCPU | key/MemoryObj/allocator 路径直接，少一层全局对象池 |
| 单机 GDS GPU <-> NVMe | LMCache GDS | GPU Direct Storage 路径明确 |
| 小规模 P2P 读 peer L1 | LMCache P2P + NIXL | 控制面较轻，直接 descriptor read |
| 多节点共享大池、跨实例 prefix 复用 | Mooncake Store | Master placement、object striping、multi-NIC、replica/lease 体系更完整 |
| 多 NIC/Rail 大对象读写 | Mooncake TE/Store | topology matrix、slice spraying、endpoint/failover 更系统化 |
| LMCache 后端就是 Mooncake | 两者组合 | LMCache 管 key/chunk/引擎还原，Mooncake 管对象和传输 |

因此更严谨的说法是：

```text
单机近端命中：LMCache 可能更低延迟。
跨节点全局对象池和多 NIC 聚合：Mooncake Store 设计更有优势。
两者不是同一层；LMCache 也可以把 Mooncake 当后端。
```

---

## 15. 结论

1. **LMCache 的变化是“换介质但保留 chunk 语义”**。  
   它把 KV Cache 组织成 `CacheEngineKey + MemoryObj`，在 CPU、disk、GDS、NIXL、Redis、S3、Mooncake 之间搬运，并尽量还原 shape/dtype/format。

2. **Mooncake direct PD 的变化是“降到地址和长度”**。  
   GPU paged tensor 变成 `(src_ptr,dst_ptr,length)`，K/V 和 layer 语义由 connector 的 region/block plan 维护。

3. **Mooncake Store 的变化是“升级为对象/副本/Slice 生命周期”**。  
   KV Cache 不只是字节，而是 object key、Replica、Segment、Slice、lease、pin 和 SSD Bucket 记录。

4. **两者不是同一层的东西**。  
   LMCache 可以管理多引擎 KV chunk 并把 Mooncake 当后端；Mooncake 则提供 RDMA 数据面与集群对象池。前者偏 engine-aware cache management，后者偏 tensor/state infrastructure。

5. **调优要看变形点，不要只看总带宽**。  
   PD 慢先查 block 地址和 region；Store 慢先查 Slice/segment/多 NIC；LMCache 远端慢先查 key、metadata、copy 次数和后端能力；SSD 慢先查对齐、bucket 和 staging。

---

## 16. 证据索引

### 参考文章

| 项 | 位置 |
|---|---|
| 用户指定微信文章 | `references/raw/wechat_1.html`（UTF-8 提取版 `wechat_1_utf8.txt`；原 `wechat_1.txt` 为 GBK 乱码） |

> **微信文章数字复核结果**（2026-09-19 源码验证）：`kMaxSliceSize = 16,777,200` 字节 ✓（= CacheLib `Slab::kSize(2^24) - 16`，types.h:427 + Slab.h:82,88）；SSD 桶 256MB/500 key ✓；`Slice {ptr, size}` ✓；`MooncakeAgentMetadata` 五字段 ✓。但 `split_k_and_v` 逻辑在 Mooncake wheel 包（`mooncake-wheel/mooncake/mooncake_connector_v1.py:702`），不在 vllm 仓库的 mooncake_connector.py（后者已演进为 TransferRegion 机制）；「160 个地址」是 80 层模型示例值而非代码常量。
| 文章主题 | Mooncake 中 KV Cache 的六次变形 |

### LMCache

| 主题 | 文件 |
|---|---|
| `CacheEngineKey` | `LMCache/lmcache/utils.py` |
| `MemoryObjMetadata` / `TensorMemoryObj` / `MemoryFormat` | `LMCache/lmcache/v1/memory_management.py` |
| GPU connector/layout | `LMCache/lmcache/v1/gpu_connector/gpu_connectors.py` |
| CPU backend | `LMCache/lmcache/v1/storage_backend/local_cpu_backend.py` |
| local disk | `LMCache/lmcache/v1/storage_backend/local_disk_backend.py` |
| GDS | `LMCache/lmcache/v1/storage_backend/gds_backend.py` |
| NIXL storage | `LMCache/lmcache/v1/storage_backend/nixl_storage_backend.py` |
| Mooncake backend | `LMCache/lmcache/v1/storage_backend/connector/mooncakestore_connector.py` |

### Mooncake

| 主题 | 文件 |
|---|---|
| vLLM direct PD | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` |
| vLLM Store worker | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` |
| Slice/ObjectDataType | `Mooncake/mooncake-store/include/types.h` |
| Replica 类型与状态 | `Mooncake/mooncake-store/include/replica.h` |
| SSD Bucket 元数据 | `Mooncake/mooncake-store/include/storage_backend.h` |
| Store 设计 | `Mooncake/docs/source/design/store/mooncake-store.md` |
| Transfer Engine | `Mooncake/docs/source/design/transfer-engine/index.md` |

---

## 附：CPU 与 GPU 之间的传输与存储设计（主报告第 3 章）

### 3.1 LMCache：MemoryObj + Paged GPU Connector + 多进程隔离

#### 3.1.1 对象模型

LMCache 的核心不是裸 byte slice，而是 `MemoryObj`：

- 包含 tensor/指针、shape、dtype、format、物理大小；
- 由 CPU pool、GDS、NIXL storage 等 allocator 创建；
- 可以作为 `batched_from_gpu()/batched_to_gpu()` 的目标；
- 可被 L1/L2/P2P 路径传递。

这个抽象让同一段上层逻辑能适配不同后端，但代价是后端必须理解 `MemoryObj` 或包一层转换。

#### 3.1.2 KV layout 单点建模

`docs/design/v1/gpu_connector/layout-invariant.md` 说明 LMCache 把 KV layout 识别收敛到 `normalize_kv_and_discover_format()`，格式由 `EngineKVFormat` 枚举和 spec class 描述：

- vLLM FlashAttention/FlashInfer；
- vLLM MLA；
- vLLM cross-layer；
- TRT-LLM HND；
- SGLang MHA/MLA；
- fused/unified KV cache。

它禁止下游随手解析 tensor shape，要求通过 spec 获取 layer、head、block、MLA、cross-layer 等事实。这是 LMCache 在多引擎生态上最重要的工程投资。

#### 3.1.3 H2D/D2H 路径

进程内 vLLM 路径：

```text
vLLM paged KV
  -> GPU staging buffer
  -> CPU MemoryObj / L1
```

反向类似。V2/V3 connector 大量使用 `tensor.copy_()` 与 non-blocking 拷贝，配合 layerwise connector 可以按层重叠计算与搬运。

MP 模式有三条路径：

1. **LMCache-driven / CUDA IPC**：worker 把 GPU tensor handle 发给 MP server，server 直接访问 worker GPU 内存。优点是延迟低、成熟；缺点是依赖 CUDA IPC 与 stream/event。
2. **Engine-driven + SHM**：worker gather/scatter 到共享内存，CPU/非 CUDA 平台可用，拷贝次数低。
3. **Engine-driven + pickle**：通用兜底，但序列化和额外拷贝开销最大。

`docs/design/v1/multiprocess/engine_driven_transfer_design.md` 给出明确拷贝数：

| 模式 | Store 拷贝 | Retrieve 拷贝 | 适用性 |
|---|---:|---:|---|
| CUDA IPC | 2 | 2 | NVIDIA，成熟，异步好 |
| SHM | 1 | 1 | 非 CUDA 或同机高吞吐，需要 `/dev/shm` |
| Pickle | 4 | 4 | 通用 fallback，开销大 |

这比很多“只要 RDMA”的宣传更诚实：跨进程和跨设备工程中，减少拷贝并不只取决于网络，还取决于引擎 layout、进程边界和设备 API。

#### 3.1.4 CPU L1 与分层

LMCache L1 常见是 CPU/DRAM pool。写路径上 L1 完成后 StoreController 异步复制到 L2；预取路径上先 lookup-and-lock L2，再 reserve L1 write buffer，再 load，最后原子转 read lock。它使用 eventfd/poll 而不是全同步阻塞，避免控制器线程阻塞。

优势：

- CPU L1 与 L2 边界清晰；
- read lock 能防止预取/搬运过程中对象被驱逐；
- 多 L2 adapter 可并行 lookup/store；
- prefix-only trim 避免加载命中片段中带洞的尾部。

劣势：

- 复制、序列化、P2P 锁与多 adapter 策略分散在多个控制器中；
- 全局缓存目录依赖 coordinator 事件流，重启后从空重建；
- 后端能力不一致：有些无容量、有些无 listener、有些 delete 是 best-effort。

### 3.2 Mooncake：注册内存、对象切片、RDMA zero-copy

#### 3.2.1 内存模型

Mooncake 的 CPU/GPU 数据面建立在注册内存上：

- DRAM/VRAM buffer 注册到 Transfer Engine；
- RDMA 场景需要 rkey/lkey；
- GPU buffer 可参与 GPUDirect RDMA；
- buffer 有 preferred NIC affinity；
- Store Client 可以把本机 DRAM/VRAM/SSD 资源贡献成 cluster segment。

这和 LMCache 的 `MemoryObj` 不同：Mooncake 的对象最终落到 cluster-managed segment，而 LMCache 的对象先留在本地 L1，再由 adapter 决定是否外移。

#### 3.2.2 Store 对象与写协议

Mooncake Store 写路径：

```text
PutStart: master 分配 replica/slice 空间
Client 写数据到各 segment
PutEnd: master 标记 COMPLETE，对象可读
```

这样避免读到 partial object。删除/读期间通过 lease 保护对象，防止 eviction/remove 与 read 竞争。文档明确这是 KV cache 对象语义，不是通用 mutable KV 数据库。

优势：

- 对象级原子可见性；
- 多副本 best-effort 分摊热点；
- slice 放在不同 segment，可与多 NIC slice spraying 匹配；
- 租约/hard pin/soft pin 可保护热点 prefix、系统提示和模型权重。

劣势：

- 集中 Master 参与空间分配，虽然数据面绕开 Master，但控制面要处理高并发与 HA；
- replication 是 best-effort，不能误当作强一致多副本数据库；
- soft pin 不进快照/OpLog，恢复后降级为普通对象。

#### 3.2.3 GPU direct path

Mooncake 支持 GPU VRAM buffer 注册与 GPUDirect RDMA。vLLM 官方 Mooncake Store 博客也解释了它不用 `cudaMemcpyAsync` 大量小拷贝，也不用 SM copy kernel，而是用 RDMA NIC 在 GPU HBM 与远端 CPU 内存间直传，不占用 SM，不需要额外 staging。

这与 LMCache 的差异很清楚：

- LMCache 进程内常见 `GPU staging -> CPU MemoryObj -> backend`；
- Mooncake Store vLLM connector 期望尽量从 paged GPU buffer 直接进入 RDMA 写/读路径；
- Mooncake Transfer Engine 能把大对象切成 slice，按 NUMA/NIC 拓扑分发。

#### 3.2.4 SSD 层

Mooncake SSD offload 的目标是把 DRAM 对象透明扩展到本地 SSD：

- heartbeat 从 master 拿 offload 列表；
- `ClientBuffer` 是 O_DIRECT 对齐的预注册 staging buffer；
- read 时 SSD -> ClientBuffer -> RDMA/GPU；
- io_uring、fixed buffer、scatter/gather、O_DIRECT、CRC、seq 恢复；
- OffsetAllocator backend 使用 4KB 对齐 value region，方便 GDS/cuFile。

LMCache 的 GDS 也做 GPU/NVMe direct path，但 Mooncake 把 SSD 与 cluster metadata/heartbeat/eviction 组合成一个统一 Store 层。后者更像分布式对象存储，前者更像引擎侧分层缓存。

### 3.3 存储结构对比

| 维度 | LMCache | Mooncake |
|---|---|---|
| 基本对象 | `MemoryObj` / `CacheEngineKey` | object + replica + slice + segment |
| L1 | CPU/DRAM、GDS、CXL/Device DAX 等可配置 | cluster segment 内 DRAM/VRAM buffer |
| 对象位置 | 本地 cache + adapter 控制外移 | Master 分配到 cluster segment |
| GPU layout | 引擎 layout 明细建模 | connector 侧注册 region，TE 按地址/length 搬运 |
| CPU copy | 可 SHM/IPC/GDS/GPU staging | 尽量注册内存 + RDMA zero-copy |
| SSD | Local disk/GDS/NIXL storage | client-owned SSD offload、DISK/DFS/NoF 层 |
| 一致性 | 本地锁/L2 lease/P2P lock，强分布式事务弱 | PutStart/PutEnd 可见性、租约、best-effort replica |
| 驱逐 | L1/L2 controller、LRU、quota、pin | master LRU、watermark、group eviction、soft/hard pin |
| 恢复 | local file/NIXL dynamic 可恢复；coordinator 目录重建 | snapshot + HA OpLog + standby catch-up |

**LMCache 的优势**：抽象边界友好，多引擎/多后端/多租户运维强；MP 模式把引擎故障与缓存故障隔离；插件化适合云厂商集成。

**Mooncake 的优势**：物理路径与 cluster capacity 是全局优化对象；大对象、多 NIC、GPUDirect、SSD offload、对象生命周期和 HA 更完整。

---
