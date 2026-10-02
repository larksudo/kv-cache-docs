---
prev: /core/prefix-hit
next: /core/async-protection
---

# H2D/D2H 接口级深潜：数据到底怎么搬

> 面试里「KV Cache 怎么从 GPU 搬到 CPU」追问到第二层就会卡壳：paged KV 为什么不能直接拷？staging buffer 是什么？CUDA IPC 到底「共享」了什么？本页全部基于 LMCache 仓库源码（`gpu_connectors.py` / `mem_kernels.cu` / `ipc_wrapper.py`）作答，面试可直接引用。

## 1. 为什么不能「直接拷」：paged KV 是散的

vLLM 的 KV Cache 物理布局是 paged 的：每层两张大表 `k_cache/v_cache[num_blocks, block_size, H, D]`，一个请求的 token 分散在**不连续的物理 block** 里（block table 指哪打哪）。而 CPU 侧的 MemoryObj 是**连续字节区**（按 chunk 组织的 `[2, L, T, D]`）。

所以「GPU→CPU 拷一份 KV」实质是 **gather/scatter 问题**：

```text
D2H（store）: k_cache[block_idx][block_offset][head][dim]  --gather-->  连续 [2,L,T,D]
H2D（load）:  连续 [2,L,T,D]  --scatter-->  k_cache[slot_mapping[i]]
```

散到什么程度：每 token 每层只取 `H×D×2` 个元素，源地址 = `block_idx × block_stride + block_offset × H×D + head_offset`——同一个 warp 里的线程访问的源地址跨 block 跳跃。**如果用 cudaMemcpy 直接对 CPU 缓冲拷，你必须把整张 paged 表（几十 GB）拷过去**——所以必须先在 GPU 侧做 kernel 级 gather/scatter 把数据「整理」成与目标布局一致的形态，再做设备间传输。这是「两跳」的第一层原因：**布局转换只能发生在 GPU 上（数据在那边），且是计算不是拷贝**。

## 2. 第一跳的接口：`multi_layer_kv_transfer` kernel

Python 侧接口（`VLLMPagedMemGPUConnectorV2.to_gpu/from_gpu`，`gpu_connectors.py`）：

```python
device_ops.multi_layer_kv_transfer(
    memory_obj.tensor,  # 目标/源：CPU MemoryObj 的连续张量（H2D 时）
    kv_cache_pointers,  # 每层 k/v cache 的 data_ptr 数组（一次性注册）
    slot_mapping[start:end],  # 本 chunk 每个 token 的物理槽位
    self.device,
    self.page_buffer_size,
    lmcache_native.TransferDirection.H2D,  # 或 D2H
    self.engine_kv_format,  # NHD/HND/MLA 等布局枚举
    block_size=...,
    head_size=...,
    block_stride_elems=...,
    skip_prefix_n_tokens=...,  # prefix cache 命中的部分跳过，防覆盖竞争
)
```

CUDA kernel（`mem_kernels.cu: load_and_reshape_flash_kernel`）内部：**每个 token 一个 block**（`token_idx = blockIdx.x`），先查 `slot_mapping[token_idx]` 得物理槽位，解出 `block_idx/block_offset`，然后 `for i = threadIdx.x; i < H*D; i += blockDim.x` 把这个 token 的 K/V 从 paged 表逐元素搬进连续目标。反向（`reshape_and_cache_back_flash_kernel`）就是同样的索引关系反过来写。几个面试可引用的设计点：

- **slot_mapping 里的 -1 是「prefix cache 命中」的哨兵**——kernel 见到负数直接 return，配合 `skip_prefix_n_tokens` 保证已缓存的块不被覆盖（多请求共享 block 的读写竞争防线）；
- **一次 kernel 调用处理所有层**：`kv_cache_pointers` 是全部层的指针数组，`key_layer_offset/value_layer_offset` 在目标连续张量里分区——层循环被打进一个 kernel，省 launch 开销（对比逐层 80 次调用）；
- **方向由枚举决定，同一个 kernel 模板**：H2D/D2H 只是 gather/scatter 的方向翻转；
- **async 与同步边界**：拷贝在专用 `store_stream` 上执行（`torch.cuda.stream(self.store_stream)`），不阻塞引擎默认流；但 `from_gpu` 在目标是 CPU 内存时会 `store_stream.synchronize()`——注释原文"for better performance, we may not want to sync for every memory object"（每对象同步是当前实现的保守点，批量场景可优化）。

## 3. 第二跳：为什么还要一个 GPU staging buffer

`from_gpu` 里有个分支（源码注释 `# kvcaches -> gpu_buffer -> memobj`）：

```python
tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]  # 预分配的 GPU 中转
multi_layer_kv_transfer(tmp_gpu_buffer, ...)  # 第一跳：paged gather 到 GPU 连续区
memory_obj.tensor.copy_(
    tmp_gpu_buffer, non_blocking=True
)  # 第二跳：GPU 连续区 → CPU pinned
```

为什么先 gather 到一个 GPU 连续缓冲、再 `copy_` 到 CPU，而不是 kernel gather 时直接写 CPU？三个原因：

1. **kernel 内直接写 CPU pinned 内存 = 跨 PCIe 的散写**。gather kernel 每 warp 写 `H×D` 个元素，目标是 CPU 时每笔都是 PCIe 事务——PCIe 4.0 ~32GB/s 的宝贵带宽被小事务打碎，且 CPU 端 uncached 写无法合并。先在 HBM 内（~3TB/s）聚成连续块，再一次性 DMA 过去，让 PCIe 只跑大块顺序传输；
2. **copy 引擎与 SM 并行**：`copy_`（cudaMemcpyAsync）由 **copy engine（DMA 引擎）执行，不占 SM**——gather kernel 在 SM 上跑下一个 chunk 时，上一个 chunk 的 PCIe 传输由 DMA 引擎同时在走，两硬件流水。直接让 SM 写远端内存则 SM 被慢速事务占住；
3. **对齐与页表安全**：pinned CPU 内存的注册、对齐由 `copy_` 路径统一处理；让任意 kernel 直接写 host 指针需要 zero-copy mapped memory 语义，开销与坑更多（且 pageable 内存根本不能被 kernel 安全写）。

**面试金句**：第一跳是「布局整理」（SM 负责，HBM 带宽），第二跳是「设备间运输」（copy engine 负责，PCIe 带宽）——两个硬件各干各的才能流水；直接一步拷等于让最快的部件去干最慢的活。

> **常见误解澄清：`copy_` 是 Python 的 copy 吗？会慢吗？**
> 不是。`tensor.copy_()` 是 PyTorch 张量的原地拷贝方法，实现在 ATen C++ 层（`aten/src/ATen/native/cuda/Copy.cu`），Python 只是调用入口（微秒级）。同 dtype + 连续 + 跨设备时它直接分派到 `cudaMemcpyAsync`——由 copy engine（DMA 引擎）执行、SM 不参与；dtype 不同/非连续时在 GPU 上起一个编译好的拷贝 kernel。真正的快慢决定因素是：① 传输量主导（1.5GB ≈ 47ms vs Python 调用几微秒）；② **目标必须 pinned**——LMCache 的 L1 用 `alloc_pinned_ptr` 页锁定，配 `non_blocking=True` 才是真异步 DMA；若目标是 pageable 内存，PyTorch 退化为「内部 staging + 强制同步」。要压 Python 层开销时的手段：`batched_from_gpu` 合并调用、CUDA Graph 消 launch、批量同步（源码注释承认的优化点）。

什么时候可以一跳：`gpu_buffer is None` 或尺寸不匹配时直接走 `memobj.tensor` 单跳（小 chunk/低频场景省掉中转的显存占用）；GDS 路径（NVMe↔GPU）和 GPUDirect RDMA（NIC↔GPU）则是**彻底绕开 CPU** 的另一维——但它们要求对齐注册（4KB/2MB），详见 [Host-GPU-RDMA 寻址与零拷贝](/core/zero-copy)。

## 4. 跨进程怎么办：CUDA IPC 到底共享了什么

MP 模式下 KV 在 vLLM worker（引擎进程）与 LMCache server（独立缓存进程）之间搬运。同一台机器上**物理显存只有一份**，IPC 的本质是**把指针变得可跨进程解引用，从而不搬数据**。

接口层（`platform/cuda/ipc_wrapper.py`）：

```python
class CudaIPCWrapper:
    def __init__(self, tensor):
        tensor = attempt_permute_to_contiguous_view(
            tensor
        )  # 非连续视图先 permute 成物理布局
        storage = tensor.untyped_storage()
        # 走 PyTorch storage IPC: share_cuda_() 得到 IPC handle（字节串）
```

- **`CudaIPCWrapper`（默认）**：用 PyTorch 的 `storage._share_cuda_()`——driver 级 `cudaIpcGetMemHandle` 生成一个 64 字节 handle；对端 `cudaIpcOpenMemHandle` 把**同一块物理显存映射进自己进程的地址空间**。跨进程传递的只有：handle 字节串 + shape/dtype/stride + CUDA event（`event_ipc_handle`，用于流同步——对端必须等生产者写完才能读）。**数据拷贝次数 = 0，跨进程传的是「钥匙」**；
- **`RawCudaIPCWrapper`（隔离容器模式）**：不依赖 PyTorch storage 语义，直接 driver-level handle——能跨完全隔离的容器，也能包装 TRT-LLM `cudaMalloc` 出来的非 PyTorch 池；
- **两个深坑（面试讲出来是区分度）**：① IPC handle **只支持 cudaMalloc 式分配，不支持 VMM**（`expandable_segments:True` 和 vLLM sleep mode 的 CuMemAllocator 都不行——源码 `_NON_IPC_MEMORY_HINT` 明说）；② **引用计数注册表**：driver 对每个 (进程, 分配) 只返回一个映射、一次 close 全部解除——同层多 tensor 共享同一分配时必须计数，漏 close 会**把已死进程的显存永久钉在物理内存里**（源码注释：a dead vLLM worker's KV pool stays resident until the server closes）；
- 同步：生产者 record event → handle 过去 → 消费者 `cudaStreamWaitEvent`——没有 event 就读 = 读到半成品。

### 三条跨进程路径的完整对比（拷贝数来源：`engine_driven_transfer_design.md`）

| 路径 | 数据流 | 拷贝次数（store/retrieve） | 适用 |
|---|---|---|---|
| **CUDA IPC** | IPC handle 映射后，server 直接读 worker 的 paged 显存（server 侧 GPU kernel gather/scatter），结果写进自己的 CPU L1 | 2 / 2 | NVIDIA，成熟，异步好；要求 /dev/shm 与同机 |
| **Engine-driven SHM** | worker 侧 kernel 先 gather 进 host 共享内存（`/dev/shm` 上的 mmap pinned 区），server 直接映射同一物理页 | **1 / 1** | 非 CUDA 或同机高吞吐；需要 L1 池放进 /dev/shm |
| **Engine-driven Pickle** | gather → 序列化 → ZMQ → 反序列化 → server 私有 L1 | 4 / 4 | 平台无关兜底（XPU/HPU/CPU） |

SHM 为什么是 1 次：`posix_shm` 分配的共享区**本身就是 pinned**（mmap 后锁页），worker 的 D2H kernel 把 paged 数据整理后直接写进这块两进程都映射着的物理页——没有「自己的缓冲再转交」的第二跳。代价：L1 池必须装进 /dev/shm（否则回退 pickle 4 次），且需要处理权限（统一用户或 umask，见[踩坑清单](/interview/cuda-systems-answers)）。

IPC 为什么是 2 次：server 侧拿到了 worker 显存的「直读权」，但 server 的 L1 是 CPU 内存——gather 结果还是要从 GPU 写到 CPU，加上 H2D 方向一次，所以 2/2。它换来的好处是 **L1 容量不受 /dev/shm 限制**、且 gather/scatter 的计算留在 GPU（比 pickle 的 CPU 序列化快得多）。

### 与 PD/跨机路径的关系（一张图收尾）

```text
同进程:   paged KV --kernel gather--> 连续 GPU --copy engine--> CPU pinned     （2 跳，本页 §2-3）
同机跨进程: IPC handle 映射（0 拷贝共享指针）→ 上面同款 kernel 在对端跑 → CPU    （数据拷 2 次）
          或 SHM: kernel gather 直写共享 pinned 页                              （数据拷 1 次）
跨节点:   paged KV --kernel gather--> 连续 GPU --GPUDirect RDMA--> 远端 GPU      （绕过 CPU，见 zero-copy 篇）
          Mooncake direct PD: data_ptr 直接进 TransferRequest，连 gather 都省    （按块连续性合并）
```

面试收束句：**「拷贝次数」的差别全部来自「布局转换发生在哪个硬件、指针能被谁解引用」**——gather/scatter 必须 SM 做、设备间运输必须 DMA 做、跨进程/跨节点能省拷贝的前提是地址可共享（IPC handle / 注册 MR），不能共享就老老实实多一跳。

## 5. 同一问题的两种答案：LMCache vs Mooncake 的单机数据面

面试官的下一问几乎必然是：「Mooncake 也搬 KV Cache，它为什么不像 LMCache 这样先 gather 到连续缓冲？谁的设计更好？」——这道题的答案是整个知识库主题的缩影。

**Mooncake 的做法：不做布局转换，用地址计算代替数据搬运。** `register_kv_caches()` 只取每层张量的 `data_ptr()` 和 `nbytes`（不解析 shape/layout）；传输时按块寻址：

```text
src_ptr = 层张量基地址 + block_id × block_len     # 每个 block 在层张量内本是连续的！
dst_ptr = 远端基地址 + remote_block_id × block_len
```

关键洞察：**paged KV 的每个 block 内部是连续的**（block_size×H×D 连续存放）——LMCache 需要的 gather 是因为它要把 256 个 token（16 个 block）聚成 chunk 语义的连续布局；Mooncake 的存储单元本来就是 block，**「整理」这一步被地址计算取代了**（`group_consecutive_contiguous()` 把连续 block 合并成大段传输，numpy 向量化完成，零数据搬运）。单机 Store 场景更直接：`TransferStrategy::LOCAL_MEMCPY` 判定同机后走线程池并行 memcpy（transfer_task.cpp:1077），连 TE 都不用。跨节点时 `data_ptr` 直接进 `TransferRequest`，GPUDirect RDMA 让 **NIC 直接读 GPU paged 显存**——gather、CPU 中转、staging 全部不存在。

**为什么会有这个差异——不是工程偏好，是分层定位的必然**：

| | LMCache | Mooncake |
|---|---|---|
| 分层定位 | 引擎侧缓存管理层（活在引擎进程边界内/旁） | 集群数据面基础设施（C++，外部进程） |
| 它「看得见」什么 | 每个引擎的 KV layout 明细（EngineKVFormat registry 是核心资产） | 只有指针和字节（「能见度递减」梯子的最底端 Slice{ptr,size}） |
| 存储单元 | chunk（256 token 连续 [2,L,T,D]，带 shape/dtype 语义） | block/字节区间（object key + 偏移） |
| 布局转换 | 必须 GPU gather（要适配 11 种引擎布局并归一化为 chunk） | **零转换**（存储布局=引擎布局，块原样搬运） |
| 额外拷贝 | +1 次 SM gather（HBM 内，快） | 0（只有地址计算） |
| 跨节点 GPU 直传 | 默认路径经 CPU（NIXL 注册 VRAM 后可 GPUDirect） | 原生 GPUDirect RDMA 从 paged 显存直传 |
| 语义操作 | chunk 粒度去重/prefix 匹配/layerwise 天然支持/Blend/压缩 | 字节级搬运不管语义；layerwise 需 vllm-ascend 的 B×R 块键区间会话补 |
| 多引擎适配成本 | 每引擎一个 adapter（V2/V3/layerwise/...） | 零适配（data_ptr 通用），但语义能力薄 |

**谁更好？按场景答**：
- **跨节点直传/大带宽池化**：Mooncake 胜——无 gather、NIC 直读显存、多 NIC 聚合，这是它作为基础设施的本职；
- **引擎内/跨引擎缓存与语义操作**：LMCache 胜——chunk 语义换来 prefix 去重、layerwise 流水、CacheBlend 非前缀复用这些「管理层」能力，这些 Mooncake 做不了（它根本不知道 token 边界在哪）；
- **收束判断**：两者正在互相渗透——MooncakeStoreConnector 用 block hash 有了前缀语义（向上长出管理层能力），LMCache 的 Mooncake L2 adapter 把物理寻址下放给 Mooncake（向下借用基础设施）。**差异的本质是「谁为 layout 买单」：LMCache 买单换来语义，Mooncake 拒绝买单换来直通**。

面试标准答法（60 秒版）：先答共同问题（paged KV 是散的，搬运必有布局适配），再答两种解法（gather 数据 vs 计算地址），然后给根因（进程边界决定 layout 可见性），最后给场景化结论。这个答题结构本身就是「分层思维」的展示。
