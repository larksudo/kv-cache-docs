---
prev: /interview/inference-answers
next: /interview/extended-answers
---

# CUDA、并行与系统题精答

> 接上篇：CUDA 与算子（第一大题族）、并行与分布式训练、网络与硬件、编程题模板、其他方向归纳。答案保持「结论先行 + 数字锚点」。

## 一、CUDA 与算子（53 家公司，270 次）

**Q1：CUDA 层级模型？内存层级？**

逻辑层级 Grid → Block → Thread（外加 Warp=32 线程的执行单元）；Block 内通过共享内存协作、可同步（`__syncthreads`），跨 Block 只能靠全局内存或 cooperative groups。内存层级由近及远：寄存器（每线程私有，最快）→ 共享内存/L1（Block 内，~100 cycle）→ L2（全 GPU）→ 全局内存（HBM，~400-800 cycle）。锚点：H100 HBM3e 约 3.35TB/s，共享内存约 128KB/SM。

**Q2：什么是 bank conflict？怎么避免？（×5）**

共享内存物理上分 32 个 bank，按 4 字节交织编址。**同一 warp 的多线程访问不同地址但落在同一 bank → 串行化**（性能掉 32 倍）；多线程读同一地址是广播不冲突。避免：padding（数组加一列错开步长）、按 bank 宽度重排访问模式、warp 内使用Swizzle。矩阵转置是经典考题——朴素转置一边合并访存一边必然 bank conflict，标准解法是 shared memory + pad。

**Q3：优化一个 CUDA kernel 从哪些维度入手？（面试第一题族，必背框架）**

先定性：**Roofline**——算术强度（FLOPs/Bytes）与机器平衡点比较，判断 compute-bound 还是 memory-bound（GEMM 不一定是计算瓶颈：小尺寸/batch 的 GEMM 是访存瓶颈）。然后按瓶颈选武器：

- **访存优化**：合并访存（coalescing，相邻线程访问相邻地址）、shared memory 复用（GEMM 分块/tile）、减少冗余读取、更大事务（向量化的 float4）；
- **计算优化**：Tensor Core（WMMA/MMA 指令）、提高占用率（occupancy = 寄存器/共享内存约束下的活跃 warp 数）、指令级并行、循环展开；
- **层次优化**：tile 尺寸扫描、split-K（K 大 M/N 小时并行 K 维度 + 原子/归约合并）、persistent kernel、double buffering 预取、warp specialization（Hopper）、TMA 异步批量搬运（省寄存器、不占 SM 的 LSU，数据可绕 L1 直达共享内存）；
- **工程**：Nsight Compute 定位（SOL 吞吐、stall 原因）、CUDA Graph 消 launch 开销、融合消除中间读写。

**Q4：GEMM 优化思路？K 远大于 M/N 怎么办？**

标准分层：tiling（block tile → warp tile → thread tile 三级）→ 数据复用（每个数据从全局读一次进共享内存被多次使用，算术强度 ↑）→ Tensor Core MMA。**K >> M/N 是 memory-bound**：split-K 把 K 维切开多 block 并行算部分和，再用 atomicAdd 或第二次 kernel 归约——用并行度换访存。另外 GEMM 常见追问：为什么按 128 对齐 tile（Tensor Core 指令形状 + 合并访存 + L2 亲和）。

**Q5：算子融合什么时候有收益？什么时候反而变慢？**

收益来自消除中间张量的全局读写（如 attention 里 QK^T+softmax+PV 融合，中间矩阵不用落 HBM——FlashAttention 的本质）。**反而变慢的场景**：融合后寄存器/共享内存超限导致占用率暴跌；两个算子本身都接近 compute-bound、无访存可省；融合打破并行度（循环依赖串行化）；JIT 编译时间在小模型上摊不平。

**Q6：手写 online softmax（高频，必练）**

安全 softmax 两遍法（先求 max 再求 exp 和）要读两遍输入。online softmax 一遍流式维护 `m = max(m, x)`、`s = s·exp(m_old−m) + exp(x−m)`，最后除 s——FlashAttention 的数学基础（分块内维护 running max 与 running sum，块间 rescale）。能口头推导 rescale 因子 `exp(m_old − m_new)` 即可。

**Q7：FlashAttention 原理？v1/v2 区别？FlashDecoding？（×16 最高频单题）**

原理：IO-aware 精确注意力——QKV 分块进 shared memory，online softmax 维持精确性，**中间注意力矩阵永不落 HBM**，HBM 访问从 O(N²) 降到 O(N²/M)。v2：**外层循环 Q、内层循环 K/V**（v1 相反）——Q 的每块只需写回一次，且前向不需要保存中间 rescale（减少非矩阵乘 FLOPs）；加上 warp 分工与更大 tile，速度约 2×。**Decoding 的问题**：Q 只有 1 行，parallelism 不足、KV 越长越糟——**FlashDecoding 把 seq_len 维度切开并行**（partial 结果），最后 combine kernel 用 online softmax 的 rescale 合并（combine 占比小，通常 <10%）。v2 外层循环选 Q 的原因：K/V 是被累加的对象（每块 Q 独立输出），以 Q 为外层输出写回连续、寄存器驻留输出。

**Q8：Hopper 新特性？TMA？**

Thread Block Allocator/Cluster（跨 SM 协作）、Warp Specialization（producer/consumer warp 分工）、**TMA（Tensor Memory Accelerator）**：硬件异步批量搬运（描述符描述一次搬运整块），不经 L1、不占 SM 线程的访存指令周期，与 mbarrier 配合实现计算/搬运流水。

**Q9：Triton/TVM 层次？**

Triton：Python DSL，block 级编程（编译器管线程编排），适合快速写 attention/layernorm 类算子；TVM：Relay（图级）→ TIR（张量级调度原语），离线编译优化。追问 Agent 生成 CUDA kernel 趋势：LLM 生成 + 编译器验证 + 性能搜索的闭环（字节/阿里已出现此题）。

## 二、并行与分布式训练

**Q10：DP/TP/PP/SP 各切什么？通信发生在哪？（×17 家公司）**

- **TP 张量并行**：权重矩阵按行/列切。按列切 GEMM 后各 rank 得到完整输出的分片，**按行切需要 AllReduce**——标准 Transformer 一层 = 列切（attention/MLP 上投影）+ 行切（下投影），两次 forward 通信（AllGather/Reduce-Scatter）+ 两次 backward；
- **PP 流水线并行**：按层切，微批次流水（**1F1B**：一个 forward 一个 backward 交替减少激活驻留；**Zero-Bubble** 把 bubble 进一步压缩到接近零——插入反向/空闲重排；DualPipe 双向流水用于 EP）；通信只在 stage 边界（P2P），带宽需求最低但 bubble 存在；
- **SP 序列并行**：Megatron 把 LayerNorm/Dropout 的激活按序列维切，与 TP 配合省激活显存；通信与 TP 复用（AllGather/ReduceScatter 换向）；
- **DP/DDP**：每卡完整模型，AllReduce 梯度——**Ring AllReduce 通信量 2(N−1)/N × 梯度大小**（与卡数无关的渐近），Tree 适合小消息低延迟。

**Q11：ZeRO 三个 Stage 的显存与通信推导？（×11）**

设参数 Ψ、优化器状态（Adam 12Ψ：fp32 参数+动量+方差混 fp16 混合计入）、梯度 2Ψ、参数 fp16 2Ψ：
- **Stage 1**：切优化器状态——省 12Ψ/N；通信量与 DDP 相同（参数 AllGather 一次）；
- **Stage 2**：再切梯度——再省 2Ψ/N；通信量不变（ReduceScatter 替代 AllReduce，总量一致）；
- **Stage 3**：再切参数——前向/反向每层临时 AllGather 参数。**论文与实现的差距考点**：Stage 2 的通信在实现里是 Reduce-Scatter + 参数广播的混合，实际通信次数略高于论文理想模型；ZeRO-Offload/Infinity 再往 CPU/NVMe 卸。

**Q12：混合精度 FP16/BF16 选择？**

BF16 指数位与 FP32 相同（8 位）——动态范围大、不易上溢下溢，训练首选；FP16 精度尾数多但范围小需要 loss scaling。KV Cache 推理侧 BF16 也是默认。锚点：TF32 是 Ampere 的矩阵乘中间格式。

**Q13：训练耗时估算？**

`总 FLOPs ≈ 6 × 参数量 × Token 数`（前向 2×+反向 4×）；`耗时 = 总 FLOPs / (卡数 × 峰值 TFLOPS × MFU)`。锚点：MFU 大集群 40-55%。

## 三、网络与硬件

**Q14：CPU vs GPU 架构本质差异？**

CPU：少数强核 + 大缓存 + 复杂控制（分支预测/乱序）——延迟优化；GPU：数万小核 + SIMT + 海量并发隐藏访存延迟——吞吐优化。GPU 适合的三个条件：数据并行、计算密集、无复杂分支。

**Q15：NVLink vs RDMA vs NVSHMEM？**

NVLink：节点内 GPU 直连（H100 900GB/s 全双工，NVSwitch 交换），延迟 ~0.1μs 级；RDMA：跨节点绕过 CPU 内核（IB NDR 400G ~50GB/s，延迟 μs 级）；NVSHMEM：基于 NVLink/RDMA 的 **PGAS 编程模型**（对称内存，直接远程指针读写）——是编程抽象 vs 物理链路的对比，不是同类。RoCE 演进五阶段与 DCQCN 详见 [物理通路](/core/transfer-paths)。

**Q16：昇腾/CANN 特点？**

达芬奇架构 Cube 单元（矩阵专用）、CANN 算力层、HCCL 通信；A2/A3 节点内 HCCS、跨机 RDMA；对齐约束多（fabric mem 1GB、HCCS 2MB）。vllm-ascend 的池化设计（物理缓冲复用 I+min(B,R)、sparse decode offload）见 [SGLang 与 vLLM-Ascend 池化设计](/ecosystem/sglang-ascend-pooling)。

## 四、编程题模板（27 家公司，70 次）

**LRU Cache（×6+3 变体，10 分钟内写完）**：哈希表 + 双向链表；`get/put` O(1)；头尾哨兵免边界判断；追问线程安全——分段锁或全局锁 + 内存序，Linux 实现可提 SIEVE（O(1) 常数更小、visited 位快速降级）。LFU 变体：频次桶 + 桶内 LRU 双链。

**单例模式（×4，含线程安全）**：C++ 用 Meyers singleton（局部 static，C++11 保证线程安全）或 DCL + `std::atomic` + acquire/release（Java 需 volatile 防指令重排）；说清 DCL 为什么要两次判空。

**高频算法题清单**（出现 ≥2 次）：合并 K 个升序链表（堆/分治，×3）、反转链表族、二叉树层序/右视图/锯齿、岛屿数量（DFS/BFS/并查集）、接雨水（双指针/单调栈）、编辑距离、滑动窗口最大值（单调队列）、TopK（快选/堆）、Hamming Weight（`n &= n-1`）、K-Coverage 区间（排序+贪心/线段树）。

**系统/工程题**：内存池实现（预分配 + free list/buddy，考虑对齐与线程安全）、多线程顺序打印（条件变量/信号量）、数组越界 coredump 排查（gdb bt + ASAN 复现 + 越界写入方地址分析）、float 比较（epsilon/相对误差/ULP）、行优先矩阵按列遍历为何慢（**cache miss**，不是带宽）。

**C++ 基础高频**：虚函数表（每个类一张、对象带 vptr、动态绑定成本）、智能指针（shared_ptr 控制块原子引用计数、weak_ptr 破环、unique_ptr 独占零开销、`enable_shared_from_this`）、四种 cast、内存对齐规则（成员按自身对齐、总大小按最大对齐取整）、移动语义（资源窃取 + 置空源）、`static` 三用途。

## 五、其他方向归纳（241 题，按类速览）

- **RAG**：检索模块方案（稠密/稀疏/混合 + 重排）、多文档冲突处理、RAG 评估（忠实度/相关性）、检索失败排查流程；
- **Agent**：记忆机制（短期对话/长期向量库）、Workflow vs Agent vs Skill 分层、MCP 协议、评估（任务完成率/轨迹）；架构设计题（工具调用链、上下文管理）——参考本知识库「 prefix cache 对 agent 场景的价值」（fresh token 仅 19%）；
- **对齐**：PPO（reward model + KL 惩罚 + clip）、DPO（直接偏好优化免 RM）、GRPO（组内相对优势，DeepSeek R1 用）；SFT 数据构造（质量>数量、多样性、指令覆盖）；
- **Diffusion/DiT**：训练加噪目标 vs 推理去噪迭代（1000 vs 40 steps 的原因——训练覆盖完整噪声谱，推理用采样器少步逼近）；DiT 推理框架与 LLM 的异同（无 KV Cache 概念但有特征缓存、patch 化序列）；Flow Matching 预测条件速度场；
- **传统 ML/系统**：GBDT vs 神经网络的表格数据优势、KNN 复杂度、TCP 握手/HTTP2 多路复用、Linux 排障命令、MySQL 索引；
- **HR/项目模式**：项目深挖五连（选型理由→个人贡献→最难点→量化收益→反思改进），提前准备每项目的数字锚点。

## 附：与其他章节的关系

本栏目是知识库的应用面：推理题的完整原理在第一/二部分，分布式在「可靠性/控制面」，池化设计在两个「内部机制深潜」，NPU 在「SGLang 与 vLLM-Ascend 池化设计」，调优参数在「验证方法与指标」。面试答题时引用具体数字（Roofline 平衡点、KV 公式、RDMA 带宽、FA IO 复杂度）是区分度所在。
