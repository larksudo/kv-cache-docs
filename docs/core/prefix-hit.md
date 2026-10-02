---
prev: /core/storage-transforms
next: /core/h2d-d2h-deepdive
---

本篇回答三个问题：Mooncake 用字节流存 KV Cache，为什么还能做 prefix cache？hash 命中后系统怎么知道该搬哪块数据、搬到哪里？MoE/FP8 与 KV Cache 是什么关系？

> 来源：V12 全文。含 Mooncake Store `PoolKey/KeyMetadata` 类定义、最长前缀命中机制与 LMCache chunk key 对照。

# Prefix Cache 命中、数据搬运与 MoE/FP8 边界

本章回答三个问题：

1. Mooncake 用字节流存 KV Cache，为什么还能做 prefix cache？  
2. hash 命中后，系统怎么知道该搬哪块数据、搬到哪里？  
3. LMCache 与 Mooncake 谁更有优势？以及 MoE/FP8 与 KV Cache 的关系。  

结论先行：

- **Mooncake Store 是“hash 命中 + 对象存在性检查 + 全局对象池”的模型**。它确实存 hash，但 hash 不是单独索引文件，而是编进 object key；命中后通过 chunk hash 找 object key，再把 object 数据搬到当前请求已分配的 vLLM paged GPU block。  
- **LMCache 是“chunk key + 多级 StorageManager 前缀匹配”的模型**。它也使用 prefix hash chain，但更强调 `CacheEngineKey -> MemoryObj -> GPU connector` 的可还原语义。  
- **Mooncake direct PD 和 Mooncake Store 不是同一条路径**。direct PD 不依赖共享 hash 池，而是按 block/region 计划做点对点搬运；Mooncake Store 才是 hash-based shared pool。  
- **MoE 不改变“模型需要 attention KV cache”的事实**。MoE 影响的是 FFN/专家计算和 token hidden state 的 dispatch/combine；Mooncake EP 的 FP8 包是 hidden-state 通信载荷，不是 Mooncake Store 对 KV Cache 做的量化。  

---

## 1. Prefix Cache 的第一性原理

Prefix cache 的假设是：如果两个请求的 token prefix 完全相同，并且模型、tokenizer、并行布局、量化格式、adapter 等都不变，那么这段 prefix 产生的 KV Cache 也相同。

因此，缓存系统真正索引的不是“这段文本”本身，而是：

```text
identity = model
         + tokenizer/revision
         + quantization/dtype/layout
         + adapter
         + parallel shard
         + prefix token hash
```

然后把这个 identity 映射到一段 KV bytes。

一个可用的 prefix cache 必须回答四个问题：

1. **如何把 token 变成稳定 key？**  
2. **如何在存储中判断哪些 key 存在？**  
3. **命中的数据在哪台节点、哪个介质、哪个对象里？**  
4. **要把数据搬到当前请求的哪个 GPU block？**  

LMCache 和 Mooncake Store 对这四个问题的实现方式不同。

---

## 2. Mooncake Store：hash 编进 object key

### 2.1 vLLM 已经生成 prefix hash chain

在 vLLM 中，请求 token 会被切成 block，并计算 block hash。这些 block hash 是链式的：

```text
h0 = H(model_context + tokens[0:block_size])
h1 = H(h0      + tokens[block_size:2*block_size])
h2 = H(h1      + tokens[2*block_size:3*block_size])
...
```

因此：

- `h0` 代表 prefix `[0, block_size)`；  
- `h1` 代表 prefix `[0, 2*block_size)`；  
- `h2` 代表 prefix `[0, 3*block_size)`；  
- 只要中间任一 token 不同，后续所有 hash 都不同。  

Mooncake Store Connector 直接复用 vLLM 传入的 `request.block_hashes`。

### 2.2 Mooncake 的 key 不是裸 hash

`vllm/.../mooncake/store/data.py` 中定义了：

```python
class KeyMetadata:
    model_name: str
    tp_rank: int
    pcp_rank: int
    dcp_rank: int
    pp_rank: int
    group_id: int
    cache_prefix: str = ""


class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str
```

稳定前缀大致是：

```text
[cache_prefix@]model_name
@tp_rank:N
@pcp:N
@dcp:N
@pp_rank:N
@group:N
```

最终 object key 是：

```text
<prefix>@<chunk_hash_hex>
```

例如概念上可以写成：

```text
tenant-a@Qwen3-32B@tp_rank:1@pcp:0@dcp:0@pp_rank:0@group:0@ab12...
```

这解决了几类冲突：

| 冲突源 | 解决方式 |
|---|---|
| 不同模型 | `model_name` |
| 不同 TP/PCP/DCP/PP shard | rank 字段 |
| 不同 hybrid attention group | `group_id` |
| 不同部署/租户命名空间 | `cache_prefix` |
| 不同 token prefix | `chunk_hash` |

注意：这个 key 目前主要是 correctness/prefix identity，不等于完整安全 authz。

### 2.3 hash 查询：`batch_is_exist`

Mooncake Store 不维护一棵完整 Radix tree，而是把每个可缓存 chunk 变成 object key，然后查询对象是否存在。

调度侧流程：

```text
request.block_hashes
  -> 按当前 block/hash 粒度展开 candidate keys
  -> MooncakeStoreWorker.lookup()
  -> store.batch_is_exist(candidate_keys)
  -> 得到存在集合
  -> coordinator.find_longest_cache_hit()
  -> 返回 usable hit_length
```

源码中的关键函数：

```python
def key_for(self, chunk_hash: BlockHash) -> str:
    return PoolKey.build_key_string(self._key_prefix, chunk_hash.hex())
```

worker 查询时：

```python
res = self.store.batch_is_exist(candidate_keys)
```

然后构造 `ExternalCachedBlockPool`，调用：

```python
_masks, hit_length = self.coord.find_longest_cache_hit(...)
```

### 2.4 “最长前缀命中”不是简单数 hash

Mooncake 还要处理 hybrid attention 和 speculative decoding：

| 情况 | 额外规则 |
|---|---|
| 多个 KV cache group | 每个 group 的对应 hash 都必须存在 |
| Full attention + SWA/Mamba | 不同 group 有不同 reachable mask |
| EAGLE / speculative decoding | 最后一个 block 可能需要丢弃重算 |
| `block_size > hash_block_size` | 用 chunk 内最后一个 sub-hash 代表该 chunk |
| 完整命中 | 通常去掉最后一个 token，保证最后 token 重新计算用于 sampling |

`_CompactChunkHashList` 的注释解释了这一点：因为 block hash 是链式的，`block_size` chunk 的最后一个 sub-hash 已经唯一代表整段 chunk 前缀，所以不需要把所有 sub-hash 拼进 key。

### 2.5 命中后怎么知道搬哪里

hit length 只回答“有多少 token 可以复用”。真正搬运还需要两个映射：

#### 映射一：hash -> Mooncake object

```text
chunk_id = start_token / block_size
block_hash = request.block_hashes[end/hash_block_size - 1]
object_key = db.key_for(block_hash)
```

#### 映射二：chunk -> 当前请求的 GPU block

调度器先为请求分配 vLLM paged GPU block。worker 拿到 `ReqMeta` 后有：

```python
req_meta.block_hashes
req_meta.block_ids
```

然后对每个命中的 chunk：

```python
key = db.key_for(block_hash)
addr, size, block_id = db.prepare_value(start, end, block_ids)
```

`ChunkedTokenDatabase.prepare_values()` 内部本质是：

```text
addr = gpu_tensor_base_addr + block_id * block_len
size = block_len * num_blocks
```

于是接收线程执行：

```python
store.batch_get_into_multi_buffers(
    keys,
    dst_addrs,
    dst_sizes,
)
```

也就是把 Mooncake Store 中的 object 数据写入当前请求已分配的 vLLM GPU paged block。

### 2.6 写路径同样由 hash 决定

写路径不是“把整个请求存成一个对象”，而是把每个可保存 chunk 变成 object：

```text
token chunks
  -> db.process_tokens()
  -> (start, end, chunk_hash)
  -> db.key_for(chunk_hash)
  -> store.batch_is_exist(keys) 去重
  -> missing keys
  -> prepare_values(..., req_meta.block_ids)
  -> store.batch_put_from_multi_buffers(keys, addrs, sizes)
```

因此 Mooncake Store 的模型是：

```text
prefix chunk hash
  -> immutable object key
  -> object replica / slices
```

它的“字节流存储”并不妨碍 prefix cache，因为对象名就是 prefix hash。

---

## 3. LMCache：chunk key + StorageManager 前缀匹配

LMCache 也使用 prefix hash chain，但抽象层级不同。

### 3.1 token -> chunk -> CacheEngineKey

`LMCache/lmcache/v1/token_database.py` 的 `ChunkedTokenDatabase` 会把 token 切成 chunk，并计算 prefix hash：

```python
token_chunks = self._chunk_tokens(tokens)
prefix_hashes = self._prefix_hash(token_chunks)

for chunk_id, hash_val in enumerate(prefix_hashes):
    start_idx = chunk_id * self.chunk_size
    end_idx = min(start_idx + self.chunk_size, total_len)
    key = self._make_key_by_hash(hash_val, request_configs)
```

`CacheEngineKey` 大致包含：

```python
CacheEngineKey(
    model_name,
    world_size,
    worker_id,
    chunk_hash,
    dtype,
    request_configs / tags,
)
```

字符串形式类似：

```text
model@world_size@worker_id@chunk_hash@dtype[@tag%value...]
```

### 3.2 lookup：连续 prefix 命中

`LMCacheEngine.lookup()` 的逻辑是：

```text
tokens/hashes
  -> token_database.process_tokens()
  -> 一串 CacheEngineKey
  -> storage_manager.batched_contains(keys)
  -> 返回连续 hit_chunks
```

默认路径要求前缀连续。例如 key 序列为：

```text
K0 K1 K2 K3 K4
```

如果只有 `K0/K1/K3/K4` 存在，LMCache 通常只使用 `K0/K1`，不会跳过 `K2`。这与 attention 对连续 prefix 的要求一致。

### 3.3 retrieve：key -> MemoryObj -> GPU

命中后，LMCache 不是直接说“这里有 N 个 token 可用”，而是继续取出对象：

```text
key
  -> storage_manager.batched_get(keys)
  -> MemoryObj(shape, dtype, format)
  -> GPU connector
  -> vLLM paged GPU KV
```

这就是 LMCache 与 Mooncake Store 的一个关键差异：

| 系统 | 命中后立刻知道 | 数据还原责任 |
|---|---|---|
| LMCache | key + MemoryObj 元数据 | LMCache/GPU connector 负责还原到引擎 layout |
| Mooncake Store | object key 存在 + replica 位置 | connector 按 vLLM block/region 拷贝 |

### 3.4 Mooncake 也可以作为 LMCache 后端

LMCache 接 Mooncake Store 时，模型是：

```text
LMCache CacheEngineKey
  -> Mooncake object key
  -> Mooncake Store
  -> Mooncake bytes
  -> LMCache MemoryObj
```

LMCache 可能使用两种模式：

1. zero-copy：LMCache 本地保存 shape/dtype/format，Mooncake 只存 bytes；  
2. metadata-embedded：`RemoteMetadata + kv_bytes` 一起写入 Mooncake。  

所以两者不是只能二选一。

---

## 4. Mooncake direct PD 与 Store 的区别

这点经常被混淆。

| 路径 | 是否用 shared prefix hash | 典型用途 | 数据从哪来 |
|---|---|---|---|
| Mooncake direct PD | 通常不是 shared pool 命中 | 当前请求 P -> D | Prefill GPU paged KV |
| Mooncake Store | 是，object key 即 chunk hash | 跨实例 prefix reuse | Mooncake object pool |
| MultiConnector | PD + Store 可组合 | 低延迟直传 + 跨实例复用 | 两条路径同时/按策略使用 |

direct PD 的搬运计划来自：

```text
remote hostname/port
remote base addr
remote block ids
local block ids
TP/PP region metadata
```

它不问“全局池里有没有这个 prefix”，而是问“对面这个请求的 paged block 在哪”。Store 的搬运计划来自：

```text
chunk hash
  -> object key
  -> object exists / replica location
  -> batch_get_into(local GPU addr)
```

---

## 5. hash 匹配的安全边界

hash 命中只能保证“大概率是同一逻辑 prefix”，不等于访问控制正确。

系统必须在 key 中至少区分：

1. model；  
2. model revision/quantization；  
3. tokenizer revision；  
4. LoRA/adapter；  
5. TP/PP/CP/DCP shard；  
6. attention type 和 hybrid group；  
7. cache salt/tenant；  
8. multimodal hash；  
9. cache schema version。  

LMCache 当前 key 包含 model、world size、worker id、chunk hash、dtype、tags。  
Mooncake key 包含 model、TP/PCP/DCP/PP rank、group id、cache prefix、chunk hash。  

但这些主要是 correctness 隔离。多租户安全还需要上层强制 tenant identity、authz、audit 和删除语义。

---

## 6. LMCache 与 Mooncake 谁更有优势？

没有绝对答案。两者优化的第一性问题不同。

### 6.1 Mooncake Store 的优势

Mooncake Store 更适合：

1. 多实例、多节点共享全局 prefix pool；  
2. 需要跨集群/跨可用区复用 agentic/multi-turn context；  
3. 希望中心化 Master 管对象生命周期、lease、pin、replica；  
4. 需要对象级 PutStart/PutEnd 可见性；  
5. vLLM-only 或 vLLM 为主的部署；  
6. 需要 fine-grained hash、hybrid attention group 和 multi-rank key namespace。  

它对 hash 管理的优势是：

```text
chunk hash -> global object key -> global existence check
```

这使得负载均衡器把请求路由到不同实例后，仍能从共享池找回 prefix。

### 6.2 LMCache 的优势

LMCache 更适合：

1. 多引擎：vLLM、SGLang、TRT-LLM 等；  
2. 多后端：LocalCPU、local disk、GDS、NIXL、Redis、S3、Mooncake、P2P；  
3. 单机或小集群内低延迟 L1/L2；  
4. 需要保留 `MemoryObj` 的 shape/dtype/format；  
5. 需要独立缓存进程与引擎故障隔离；  
6. 需要跨后端统一 lookup/prefetch/pin/evict 接口。  

它对 hash 管理的优势是：

```text
chunk hash -> CacheEngineKey
          -> 同一套 StorageManager 多级 lookup
          -> MemoryObj
          -> engine connector
```

这让 LMCache 在本地/近端命中时路径更短，也更容易接入已有存储系统。

### 6.3 场景化结论

| 场景 | 更强的一方 | 原因 |
|---|---|---|
| vLLM 多实例 agentic shared pool | Mooncake Store | 全局 object key、跨实例命中、对象生命周期更完整 |
| 单机 CPU/disk offload | LMCache | LocalCPU/local disk/GDS 路径直接 |
| 多引擎统一缓存 | LMCache | engine layout registry 和多后端抽象更通用 |
| RDMA 大带宽 PD direct | Mooncake TE | multi-NIC、topology、failover 更完整 |
| 云厂商强治理 | Tair/Mooncake 类系统 | 租户/审计/HA 能力更接近企业存储 |
| Mooncake 作为远端后端 | 组合 | LMCache 管 chunk/多引擎，Mooncake 管分布式对象池 |

### 6.4 简短回答

如果问题是“谁能做更好的跨实例 prefix cache pool”，当前证据更支持 **Mooncake Store**。  
如果问题是“谁能把不同引擎、不同介质、不同后端统一管理”，当前证据更支持 **LMCache**。  
如果问题是生产系统怎么选，最佳答案通常是 **按场景组合**，而不是二选一。

---

## 7. MoE 阶段为什么还需要 KV Cache？

这是一个常见混淆：MoE 并没有取消 attention。

一个 decoder layer 通常包含：

```text
input
  -> attention
  -> FFN / MoE
  -> output
```

KV Cache 存的是 attention 需要的历史 K/V，而不是 MoE 专家的权重或输出。MoE 改变的是 FFN 部分：每个 token 只路由到部分专家。

因此：

1. 模型仍然是 decoder transformer，仍然有历史 token；  
2. 当前 token 的 attention 仍需要前面所有 token 的 K/V；  
3. decode 阶段如果删除 KV Cache，就要重跑整段 prefill；  
4. MoE 只影响“这一层的 FFN 怎么算”，不影响“attention 需要历史 K/V”。  

所以 MoE 模型仍然需要 KV Cache。

---

## 8. MoE dispatch 里的数据不是 KV Cache

MoE 的专家并行流程是：

```text
local token hidden states
  + router/topk expert ids
  -> dispatch 到 expert rank
  -> expert FFN 计算
  -> combine 回 token owner
```

这里的 payload 是：

```text
token hidden state
topk expert id
topk weight
expert output
```

不是：

```text
K[token, layer, head, head_dim]
V[token, layer, head, head_dim]
```

Mooncake EP 文档也明确 `dispatch()` 的输入是 `x: [num_tokens, hidden]` token hidden states 和 `topk_idx`。combine 再把 expert 输出送回 token owner。

所以：

```text
MoE dispatch payload = hidden-state communication
KV Cache = attention history state
```

两者都可能是张量、都可能走 RDMA，但生命周期和复用语义不同。

---

## 9. 为什么 KV Cache 可以量化？

KV Cache 是浮点张量，本质上可以量化。其可行性来自三点：

1. attention 对 K/V 的小幅数值扰动有一定容忍度；  
2. 可以按 token/channel/head/layer/group 保存 scale；  
3. 很多 attention kernel 已支持 FP8/BF16/INT8 KV 输入。  

例如 FP8 KV 常见形式是：

```text
quantized_k = fp8(k / scale_k)
scale_k    = per-token/per-channel/per-group scale
quantized_v = fp8(v / scale_v)
scale_v    = ...
```

attention kernel 使用 quantized K/V 时必须知道：

1. dtype：FP8 E4M3/E5M2、INT8 等；  
2. scale 布局；  
3. quantization group；  
4. head/layout；  
5. 是否需要 dequantize。  

因此，**量化改变的是数值表示，不改变 prefix identity**。

这一点非常重要：

```text
prefix hash = H(tokens + model/context metadata)
```

而不是：

```text
prefix hash = H(quantized KV bytes)
```

否则 BF16、FP8、INT8 版本会被当成不同 prefix，无法跨精度复用，也无法保证语义一致。

---

## 10. Mooncake 在这里做了什么？

要分开三个 Mooncake 组件。

### 10.1 Mooncake Store：不主动量化 KV Cache

Mooncake Store 的对象有：

```cpp
enum class ObjectDataType : uint8_t {
    UNKNOWN = 0,
    KVCACHE = 1,
    TENSOR = 2,
    WEIGHT = 3,
    ...
};
```

这只是对象类型标签。Store 的数据路径看到的是 bytes/Slice/object。它不会自动把 BF16 KV 转成 FP8，也不会理解 attention kernel 需要的 scale。

如果上层 connector 已经得到 FP8 KV，Mooncake Store 可以把它当字节对象存；但量化必须由引擎/attention backend/connector 负责。

### 10.2 Mooncake Transfer Engine：搬运已量化字节

如果 connector 给它的是 FP8 KV bytes，它就搬运 FP8 bytes；如果是 BF16 KV bytes，就搬运 BF16 bytes。Transfer Engine 不改数值语义。

### 10.3 Mooncake EP：FP8 用于 hidden-state dispatch

Mooncake EP 的 `dispatch(..., use_fp8=True)` 会把 token hidden states 打包成 FP8 数据和 scales：

```text
x: [num_tokens, hidden]
topk_idx
  -> FP8 payload + FP32 scales
  -> expert compute
  -> combine
```

这不是 KV Cache 的量化。它是 MoE expert-parallel 通信的量化。

参考博客中的“EP FP8 包”更准确的定位是：

```text
MoE dispatch/combine 的 hidden-state 载荷
```

而不是：

```text
Mooncake Store 的 KV Cache 持久化格式
```

---

## 11. 如果要做 KV Cache 量化，谁负责什么？

| 职责 | 负责方 |
|---|---|
| 决定哪些 layer/head 可以量化 | model / serving engine / quantization policy |
| 计算 scale | quantization kernel / attention backend |
| 选择 FP8/INT8 kernel | vLLM/SGLang/TRT-LLM 等引擎 |
| 保存 dtype/scale/layout metadata | cache manager / connector |
| 搬运 quantized bytes | LMCache / Mooncake TE / Mooncake Store |
| 防止不同量化格式错误复用 | prefix key / namespace / schema version |
| 质量评估 | model owner / eval pipeline |

Mooncake Store 的角色是安全搬运和生命周期管理；不是替模型决定量化。

---

## 12. 量化 KV 的风险

| 风险 | 说明 |
|---|---|
| 质量退化 | 某些 layer/head 对 K/V 扰动更敏感 |
| kernel 不兼容 | producer 保存 FP8，consumer attention backend 不支持 |
| scale 丢失 | 没有正确 scale 就是纯乱码 |
| layout 不匹配 | per-token/per-channel/per-group scale 布局不同 |
| 跨精度复用错误 | BF16 命中 FP8 或反向，可能语义不一致 |
| key 设计错误 | 只用 token hash 但未绑定量化和 schema 版本 |
| hash 与值混淆 | prefix hash 应来自 token/context，不应来自量化后的 KV bytes |

生产上应把量化 KV 视为新的 cache schema：

```text
model
  + tokenizer revision
  + adapter
  + quantization scheme
  + scale layout
  + attention backend
  + prefix hash
```

---

## 13. 端到端例子

### 13.1 Mooncake Store 命中并搬运

```text
新请求 tokens:
  [system prompt 4096][user question 128]

vLLM block hashes:
  h0, h1, ..., h263

Mooncake worker lookup:
  candidate_keys = [
    model@tp0@pcp0@dcp0@pp0@group0@h0,
    model@tp0@pcp0@dcp0@pp0@group0@h1,
    ...
  ]
  exists = store.batch_is_exist(candidate_keys)

coordinator:
  longest hit = 4096 tokens

scheduler:
  为剩余 128 token 分配 GPU block
  为复用 4096 token 也分配目标 block id

worker:
  for each hit chunk:
      key = prefix + chunk_hash
      dst_addr = gpu_tensor_base + block_id * block_len
  store.batch_get_into_multi_buffers(keys, dst_addrs, dst_sizes)

vLLM:
  只对最后 128 token 做 incremental prefill
```

### 13.2 LMCache 命中并搬运

```text
tokens
  -> chunk hashes
  -> CacheEngineKey0..N
  -> StorageManager.batched_contains(keys)
  -> longest prefix hit
  -> StorageManager.batched_get(keys)
  -> MemoryObj list
  -> GPU connector batched_to_gpu()
  -> vLLM paged GPU KV
```

如果 LMCache 的 L2 是 Mooncake：

```text
CacheEngineKey
  -> Mooncake object key
  -> Mooncake object bytes
  -> LMCache MemoryObj
  -> vLLM paged GPU KV
```

---

## 14. 最终判断

1. **Mooncake Store 是 hash-based prefix cache。**  
   它把 vLLM block/prefix hash 编进 object key，用 `batch_is_exist()` 做存在性查询，再用 coordinator 找最长可用前缀。

2. **hash 命中后靠两个映射搬运数据。**  
   `hash -> object key` 告诉系统去存储池哪里取；`chunk -> current GPU block id/address` 告诉系统写到当前请求哪里。

3. **LMCache 与 Mooncake 是不同层级的优势。**  
   Mooncake Store 更像全局分布式对象/prefix pool；LMCache 更像多引擎、多介质、多后端的缓存管理层。

4. **MoE 仍然需要 KV Cache。**  
   MoE 改变的是 FFN/专家计算，不取消 attention。专家并行 dispatch 的是 hidden states，不是 KV Cache。

5. **KV Cache 可以量化，但 Mooncake 不是量化器。**  
   Mooncake Store 可以存已量化 KV bytes，Mooncake EP 可以对 hidden-state dispatch 做 FP8，但 KV 量化和 scale/兼容性由模型/引擎/attention backend 决定。

---

## 15. 证据索引

### Mooncake Store prefix cache

| 主题 | 文件 |
|---|---|
| chunk hash / key metadata / object key | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py` |
| 最长前缀命中 / partial hash / hybrid group | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py` |
| scheduler lookup | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py` |
| `batch_is_exist()` / 写入 / 读取 / lookup server | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` |
| lookup wire protocol | `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/protocol.py` |

### LMCache prefix cache

| 主题 | 文件 |
|---|---|
| token -> chunk hash -> key | `LMCache/lmcache/v1/token_database.py` |
| `CacheEngineKey` | `LMCache/lmcache/utils.py` |
| lookup / retrieve / GPU restore | `LMCache/lmcache/v1/cache_engine.py` |
| MemoryObj 元数据 | `LMCache/lmcache/v1/memory_management.py` |

### Mooncake EP / FP8

| 主题 | 文件 |
|---|---|
| EP dispatch/combine 设计 | `Mooncake/docs/source/design/mooncake-ep.md` |
| `use_fp8=True`、FP8 payload 和 scales | `Mooncake/docs/source/design/mooncake-ep.md` 第 133-156 行附近 |
| EP dispatch API | `Mooncake/mooncake-ep/include/mooncake_ep_buffer.h` |
| MoE 定位 | `Mooncake/README.md` |

### 参考文章

| 项 | 位置 |
|---|---|
| Mooncake 六次变身文章 | `references/raw/wechat_1.html` / `references/raw/wechat_1.txt` |

## 附：跨位置复用与引擎边界（增量）

> 来源：ForceInjection 系列 RoPE 与 Prefix Caching 篇，证据等级 B/C。

三种跨位置复用方案的位置：vLLM APC 与 SGLang 默认 RadixAttention 属「方案 A（位置限制匹配）」；MLA 的 KV latent 不含位置信息，天然支持「方案 B（nope 哈希 + 重算 RoPE）」；**LMCache CacheBlend 是目前少数工业级走通「方案 C（Rotary Correction）」的方案**——利用 RoPE 旋转矩阵正交可加性，从位置 m 校正到 m' 只需乘旋转差矩阵，支持跨位置/跨片段复用（RAG 拼接场景）。SGLang 跨位置复用需集成 CacheBlend。

另一个跨引擎事实：量化与 prefix caching 兼容（hash 基于 token ID 不依赖 K 值），但 vLLM Connector API 不内置精度转换——跨精度复用需 producer/consumer 同 dtype，connector 实现者需在传输层自行处理。
