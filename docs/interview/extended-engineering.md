---
prev: /interview/extended-rag-agent
next: /appendix/evidence
---

# 扩展题库 · ML/CV、工程基础与开放题（逐题精答）

> 约 100 题逐题精答。OS/网络/云原生/DB 是百度、腾讯、蔚来系的重仓；ML/CV 一题一答；开放题给答题框架。

## A. 传统机器学习

**A1. Logistic 回归原理与损失推导？（×2）**
线性 logit `z=wᵀx+b` 过 sigmoid 得概率 `p=σ(z)`；从最大似然推出交叉熵 `L=−[y·log p+(1−y)·log(1−p)]`（伯努利分布负对数似然）。梯度形如 `∇=(p−y)x`——干净且不会像 MSE+sigmoid 那样在饱和区梯度消失。

**A2. 分类任务为什么不用 MSE？**
两个原因：① MSE+sigmoid 的梯度含 `σ'(z)` 因子，z 大时饱和趋零、学不动；交叉熵梯度恰好约掉这项。② MSE 对「错得很自信」的惩罚不足，交叉熵按概率距离指数惩罚。另外 MSE 假设高斯噪声，分类是伯努利，分布假设就错了。

**A3. 过拟合方案？L1/L2 区别？L2 导数？**
方案：正则、dropout、early stopping、数据增强、降模型容量、交叉验证选参。L1（`λΣ|w|`）：导数恒 ±λ，能把权重压到精确零——稀疏特征选择；几何上是菱形等高线与损失椭圆交点在坐标轴上。L2（`λΣw²`）：导数 `2λw`——每步按比例缩小（weight decay），权重趋小不趋零、平滑。L1 角点稀疏、L2 圆边平滑是标准答法。

**A4. GBDT 工作机制？**
加法模型 + 逐棵拟合残差（更准确说是负梯度）：`F_m(x)=F_{m-1}(x)+η·h_m(x)`，每棵新树学「当前模型还差多少」，学习率 η(0.1) 收缩防过拟合。与随机森林对比：GBDT 是 boosting（串行、降偏差、树浅），RF 是 bagging（并行、降方差、树深）。

**A5. 随机森林原理？**
Bootstrap 采样（行）+ 特征随机子集（列）训多棵深树，投票/平均。两个随机带来方差下降且树间去相关；OOB（袋外样本）免费做验证。

**A6. KNN 流程与复杂度？**
算待分类点与全部训练点的距离 → 取 top-k → 多数投票。训练零成本，**推理 O(N·d)**——这是它与深度模型相反的代价结构（训练贵推理便宜 vs 训练零推理贵），KD-tree/ball-tree 加速到 O(logN)（低维有效）。k 小过拟合、k 大欠拟合。

**A7. 树模型 vs 线性模型？**
树：自动特征交互、无需归一化、可解释（特征重要性）、处理混合类型；难外推（台阶函数）、高方差易过拟合。线性：外推平滑、系数即解释、推理极快；需手工特征工程、建模不了交互（除非显式交叉项）。表格数据 XGBoost 常胜 DL 是行业经验。

**A8. 特征筛选方法？**
三大类：**过滤式**（统计量与标签相关性：方差、卡方、互信息、相关系数——快但忽略特征间组合）、**包裹式**（用模型效果搜子集：RFE 递归消除——准但贵）、**嵌入式**（训练时自带选择：L1、树模型重要性）。工程顺序：先过滤砍一半再包裹精筛。

**A9. 正负样本不均衡怎么办？**
数据：欠采样多数类（配合集成防丢信息）、过采样少数类（SMOTE 插值）。损失：class weight、Focal Loss（降易样本权重）。指标：**别用 accuracy**——看 PR-AUC/召回率（1:99 时全预测多数类 accuracy 99% 但无意义）。

**A10. 时间序列怎么 ML 特征筛选 + 规则建模？**
特征：滞后值、滑窗统计（均值/方差/斜率）、周期特征（小时/星期 one-hot）、外部变量；筛选用互信息或树模型重要性。规则层：阈值告警、趋势突变检测、业务边界（如「超过历史 P95 即异常」）——ML 出分数、规则做决策是工业常态。毫秒级精度追问：改用更细粒度窗口 + 在线特征实时计算（流式聚合），代价是计算成本与抖动。

## B. 计算机视觉

**B1. 目标检测基本原理与主流方法？**
回归「类别+边界框」。两阶段（Faster R-CNN：RPN 提候选→RoI 精修，准但慢）；单阶段（YOLO/SSD：密集锚框直接回归，快）；DETR 系（Transformer 端到端集合预测，免 NMS 免锚框）。

**B2. YOLO v1/v2/v3 演进与 vs SSD？**
v1：单网络网格直接回归（开创单阶段）；v2：锚框+BN+高分辨率微调；v3：多尺度预测（FPN 三头）+Darknet53。vs SSD：SSD 多层特征图各挂预测头但浅层语义弱；YOLOv3 的 FPN 融合让小目标显著改善。YOLOv5：CSP 结构、PAN 路径聚合、autoanchor、工程化（PyTorch 原生、一键训）。

**B3. YOLO 推理加速与部署？**
模型侧：剪枝（通道/层）、量化（INT8 PTQ）、蒸馏（大模型教小模型）、换轻骨干（MobileNet）；部署侧：TensorRT/ONNX Runtime 图优化+算子融合、batch 化、输入分辨率下调。流程：pt→onnx→trt engine（标定 INT8）→C++ 推理服务。

**B4. 多目标跟踪与 ByteTrack 流程？**
Detection + ReID + 数据关联。ByteTrack 关键创新：**低置信度检测框不丢弃**——先用高置信度与轨迹匹配（卡尔曼预测 + IoU/ReID 相似度匈牙利算法），未匹配的轨迹再与低置信度框二次匹配（低分往往是遮挡中的目标，丢掉就断轨迹）。

**B5. BEV 三代演进？LSS vs BEVFormer 算力？**
第一代 LSS（Lift-Splat-Shoot）：多相机图像显式「抬升」到 3D 空间再拍扁到 BEV——逐像素深度分布，计算冗余大；第二代 BEVFormer：Transformer 在 BEV 空间做 spatial/temporal attention 查询采样——精度高但 attention 计算重；第三代 Diffusion-based（如 BEVDiff）：生成式建模 BEV，长尾与不确定性更好。算力：BEVDet 系（LSS 路线）通常低于 BEVDFormer（dense attention 是大头）。

**B6. MobileNet/GhostNet/SE？**
MobileNet v1 深度可分离卷积（空间与通道分离，计算降 1/k）；v2 倒残差（先升维再线性瓶颈防信息损失）+ ReLU6；v3 NAS 搜结构 + squeeze-excite。GhostNet：卷积输出中许多特征图相似——只卷一半，另一半用廉价线性变换（如 depthwise）生成。SE 模块：全局池化→FC 学通道权重→重标定（通道注意力鼻祖）。

**B7. RT-DETR 相比 DETR 的改进？**
DETR 收敛慢（500 epoch）、小目标弱——RT-DETR：高效混合编码器（只对少量候选 query 做解码）、IoU-aware 查询选择（先验更好的初始 query）、去掉 NMS 的同时做到实时——首个实时端到端检测器。

**B8. CLIP 原理与推理流程？**
双塔：image encoder（ViT）与 text encoder 分别嵌入到同一空间，对比学习（InfoNCE）拉近配对图文。推理两种用法：零样本分类（类别文本模板 embedding 与图像 embedding 算余弦取最大）；检索（图文互查）。Zero-shot 能力来自 4 亿图文对的规模。

**B9. 卡尔曼滤波原理？**
两步循环：**预测**（`x=Ax+Bu, P=APAᵀ+Q`——按运动模型外推）与**更新**（用观测 z 修正：增益 K 权衡预测方差 P 与观测方差 R）。直觉：预测和观测各有一个不确定度，K 决定信谁多一点。应用：跟踪里的状态平滑、导航、传感器融合。

**B10. 全局池化位置与作用？**
通常在 backbone 尾部、分类头之前：把 H×W 特征图压成单向量（GAP），作用是聚合空间信息、参数归零（替代大 FC）、对输入尺寸不敏感（支持多尺度测试）。

**B11. 类别数与数据集规模对检测的影响？**
类别多：分类头线性变大、长尾严重（尾部类样本不足——重采样/重加权）、NMS 更耗时；数据少：过拟合+锚框/先验统计不准——迁移学习、数据增强、少样本（冻结骨干只训头）。

**B12. 梳理 CV 发展脉络？**
AlexNet（2012，ReLU+Dropout+GPU，深度学习起点）→ VGG（小核堆深度）→ GoogLeNet（Inception 多尺度）→ ResNet（残差连接解深网退化，里程碑）→ 注意力时代（ViT：图像即 patch 序列）→ CLIP/DiT/多模态。激活函数线：ReLU→LeakyReLU→GELU/SwiGLU。一句话主线：**从手工特征到深度特征到架构即先验**。

## C. 操作系统

**C1. 进程 vs 线程 vs 协程？（×3）**
进程：资源分配单位（独立地址空间/文件描述符），切换开销大（页表/缓存重建）；线程：调度单位（共享进程地址空间），切换只换寄存器栈；协程：**用户态**调度的轻量执行流（切换不进内核，ns 级），单线程内并发 IO 但占不满多核（配线程池用）。用户态线程即协程概念的 OS 教科书叫法。选择：隔离需求→进程；计算并行→线程；高并发 IO→协程。

**C2. IPC 方式有哪些？（×2）**
管道（父子单向）、命名管道 FIFO、消息队列（内核态中转）、共享内存（**最快**——同一物理页映射两进程，零拷贝，配信号量同步）、信号量（同步原语）、socket（跨机通用，最慢但最通用）、信号（异步通知）。KV Cache 领域联系：LMCache MP 的 SHM 模式就是「共享内存 + pinned」的工程化（见 [H2D 深潜](/core/h2d-d2h-deepdive)）。

**C3. 虚拟地址空间布局？**
x86-64 Linux 自低到高：text（代码）→ data/bss（已初始化/未初始化全局）→ heap（向上长，malloc）→ mmap 区（共享库/映射文件）→ stack（向下长，8MB 默认）→ 内核空间（高 128T）。机制：多级页表翻译 + TLB 缓存 + 缺页中断按需分配——虚拟化换隔离与灵活。

**C4. 内存管理核心内容？**
分页（页表/大页——TLB miss 少，KV 系统用 2MB 大页的原因）、按需分配（缺页中断）、写时复制 COW（fork 快、block 共享）、交换（swap——pinned 内存被锁定的原因：DMA 期间不能被换出，见 [零拷贝篇](/core/zero-copy)）、OOM killer。

**C5. 文件管理机制？**
inode（元数据与数据块指针分离）、目录树、page cache（读写缓存——O_DIRECT 绕过它的原因：KV offload 自己管缓存，双重缓存浪费内存）、VFS 抽象层、日志式文件系统（ext4）防崩溃不一致。

**C6. Linux 启动过程？**
BIOS/UEFI 自检 → bootloader（GRUB）加载内核 → 内核初始化（驱动/挂载 rootfs）→ 启动 1 号进程 systemd → 按 target 拉起服务。追问点：单用户模式救援、内核参数（如 hugepages 配置就在 bootloader 传参）。

**C7. Cache 层级与替换策略？**
L1（每核，~4 周期，分指令/数据）→ L2（每核，~12 周期）→ L3（全核共享，~40 周期）→ 内存（~200 周期）。替换：LRU（经典）、随机（实现简单抗扫描）、PLRU、伪 LRU。行优先矩阵按列遍历慢的原因就是 cache line（64B）失效——**劣化的指标是 cache miss rate 不是带宽**。

**C8. 零拷贝原理与应用？**
传统读文件发送：磁盘→内核页缓存→用户缓冲→socket 缓冲→NIC（4 次拷贝 4 次切换）。零拷贝：`sendfile`（页缓存直达 NIC scatter-gather，2 次拷贝 0 次用户态切换）、`mmap+write`（映射共享减一次）、splice（管道中转）。应用：Nginx/Kafka 大文件吞吐。与 KV 系统的联系：见 [零拷贝篇](/core/zero-copy)——RDMA MR 注册是零拷贝的网络版。

**C9. 高频命令与调试？**
磁盘：`df -h / du -sh / iostat -x / lsblk`；文本替换：`sed -i 's/a/b/g' file`；进程：`ps aux / top / pidstat`；网络：`ss -tlnp / netstat / tcpdump`；调试：`gdb`（bt/full/断点）、`strace`（系统调用追踪）、`ltrace`（库调用）、`perf`（热点采样）、`lsof`（文件句柄）、`dmesg`（内核日志，OOM 在这看）。coredump 数组越界排查：gdb 加载 core → bt 找崩溃栈 → 检查越界写入方的地址计算（C++ 系加分：先 ASAN 复现更直接）。

**C10. 通信时延影响因素与缓解？**
因素：传播（物理距离，光速上限）、传输（带宽×消息大小）、排队（拥塞）、处理（协议栈/拷贝）。缓解： locality（同机架/同 NUMA——本库 PD 同 Leaf 原则）、批量化（摊薄 RTT）、RDMA 绕内核、压缩、流水重叠（计算藏传输——本库四条轴的重叠轴）。

## D. 网络

**D1. TCP 三次握手与四次挥手？**
握手 SYN→SYN+ACK→ACK（三次够了：双方各确认一次收发能力；两次服务端无法确认客户端能收）。挥手多一次是因为**半关闭**：被动方收到 FIN 后可能还有数据要发，ACK 与自己的 FIN 分开发。TIME_WAIT 2MSL：防最后 ACK 丢失重传的 FIN 无处应答 + 让旧连接报文自然死亡。

**D2. TCP 可靠传输机制？**
四件套：**序号+确认**（字节流编号，累计 ACK）、**超时重传**（RTO 自适应）、**校验和**、**拥塞控制**（慢启动/拥塞避免/快重传/快恢复）；外加流量控制（滑动窗口，接收方驱动）。RDMA 为什么丢包敏感可接这题：RDMA 把这些全卸到硬件且假设无损网络——丢包代价是 100GB/s 掉到 10GB/s（见 [物理通路](/core/transfer-paths) RoCE 附录）。

**D3. TCP/UDP/HTTP/HTTPS 对比？**
TCP 面向连接可靠有序（握手、重传、慢）；UDP 无连接（快、可丢、实时音视频/QUIC 底座）；HTTP 应用层（请求响应、无状态）；HTTPS = HTTP + TLS（非对称握手换对称密钥，约 1-2RTT 额外，TLS1.3 降到 1）。选型看：可靠性要求、延迟敏感度、连接开销。

**D4. HTTP/2 vs 1.1？**
① 二进制分帧（解析快）；② **多路复用**（一个 TCP 连接并行多流——解决 1.1 队头阻塞 at HTTP 层）；③ 头部压缩 HPACK；④ 服务端推送。遗留问题：TCP 层队头阻塞仍在（一个丢卡全连接）——HTTP/3 换 QUIC（UDP）解决。

**D5. TCP/IP 各层职责？**
应用（HTTP/DNS——语义）→ 传输（TCP/UDP——进程到进程、可靠性）→ 网络（IP——主机到主机、路由）→ 链路（以太网——相邻节点、MAC）。ARP 在 2.5 层（IP→MAC 解析）。跨层联系：RoCEv2 把 RDMA 载荷直接封装在 UDP/4791 上——绕过 TCP/IP 协议栈是低延迟的根源。

## E. 云原生与容器

**E1. K8s 架构与核心概念？**
控制面：apiserver（唯一入口）、etcd（状态存储）、scheduler（绑节点）、controller-manager（调谐循环）。节点：kubelet（Pod 生命周期）、kube-proxy（服务转发）。Pod = 最小调度单元（一组共享网络/存储的容器）。

**E2. Pod 创建流程？CSI/CNI 何时介入？**
kubectl→apiserver（写 etcd，Pod Pending）→ scheduler watch 选节点绑定 → kubelet watch 到本节点 → 创建 sandbox（**CNI 在此分配 Pod IP**）→ 挂载卷（**CSI 在此**）→ 启动容器探活 → Running。即 CNI 在网络沙箱阶段、CSI 在卷挂载阶段，都在容器启动前。

**E3. Informer 机制？**
List-Watch 模式：启动时 List 全量 + 之后 Watch 增量，全部存本地缓存（Store）；事件分发 DeltaFIFO → 回调 handler。控制器读本地缓存而非打 apiserver——**把查询压力变成 O(1) 内存读**，这是 K8s 能撑数千节点的关键设计。

**E4. Docker 原理与 namespace？**
容器 = 进程 + 六种 namespace 隔离（mnt 文件系统、pid 进程号、net 网络栈、ipc、uts 主机名、user）+ **cgroups 资源限制**（CPU/内存/IO）。容器通信：bridge（默认 veth+网桥+NAT）、host（共享宿主栈）、overlay（跨主机 VXLAN）、none。镜像 = 分层联合文件系统（overlayfs）。容器没有 VM 的 hypervisor——共享内核是轻的原因也是隔离弱的原因。

**E5. Java Full GC 排查？（跨界题）**
`jstat -gcutil` 看频率 → `jmap -histo`/dump 找大对象 → 老年代增速与晋升阈值（大对象直入老年代/元空间泄漏/内存不足频繁晋升）。解决：调堆与新生代比例、避免大缓冲、G1/ZGC 换代。附带 Java 装箱题：`Integer a=567,b=567; a==b` 为 **false**——IntegerCache 只缓存 -128~127，567 各自新对象，== 比较引用；用 equals 或 intValue。

## F. 数据库

**F1. EXPLAIN 看什么？怎么判断走没走索引？**
关键字段：**type**（访问类型：system>const>eq_ref>ref>range>index>ALL——ALL 即全表扫描）、**key**（实际用的索引，NULL 即没用上）、**rows**（预估扫描行数）、**Extra**（Using index 覆盖索引好事；Using filesort/temporary 要优化）。

**F2. 联合索引 (a,b,c) 与最左前缀？**
索引按 a 排序、a 相同按 b、再按 c——查询条件必须有 a 才能用（a,c 可用 a 部分；无 a 则全失效）。范围查询（a>x）后停止匹配后续列。题目「查询条件 b,c」：用不上索引——要么改查询要么加 (b,c) 索引。

**F3. 事务与 MVCC？**
ACID 中 MVCC 服务 Isolation：InnoDB 每行存隐藏列（创建/删除事务号）+ undo 链，读时按 Read View 判断可见性——**读不加锁读写不阻塞**。RR 级别用事务开始时的快照，RC 用每条语句的新快照。写冲突仍靠锁（当前读）。

**F4. 主从复制强一致？**
默认异步（主提交即返回，从库可能落后）——主崩丢已确认事务。半同步（至少一个从收到 relay 才返回）；真正强一致需组复制/Paxos 类（MySQL Group Replication、TiDB Raft）或牺牲可用性（全同步）。追问可提：GTID 让从库重放幂等。

**F5. SQL 执行流程（MiniOb 式）与最慢环节？**
解析（词法/语法→AST）→ 语义/绑定（表列校验、权限）→ 优化（逻辑改写、**基于代价的物理计划选择**——JOIN 顺序、索引选择，通常最耗时也最值钱）→ 执行器（火山模型迭代器）→ 存储引擎。性能关键环节：优化器计划选择与缓冲池命中。

## G. Python/工具

**G1. 深浅拷贝？GIL？装饰器？（高频组合）**
浅拷贝（copy/list()）复制容器不复制元素（嵌套对象共享）；深拷贝 deepcopy 递归复制。GIL：CPython 全局锁使同一时刻仅一线程执行字节码——CPU 多线程无效，IO 多线程有效（锁在 IO 时释放），多核用进程。装饰器：接受函数返回函数的高阶函数，@ 语法糖；带参装饰器三层嵌套；functools.wraps 保元数据。

**G2. Python 内存管理与循环引用？**
引用计数为主（即时回收）+ 分代 GC 为辅（0/1/2 代，阈值触发标记-清除）——**循环引用引用计数无解**（互指计数不归零），标记-清除从根对象可达性遍历，不可达整体回收（含环）。

**G3. 局部变量怎么返回外部使用？**
返回值拷贝（小对象）/返回引用（列表字典——注意可变对象逃逸后的生命周期）/闭包捕获/nonlocal。陷阱：循环里 lambda 捕获的是变量名不是值（late binding）。

**G4. Git 五连。**
fetch+checkout vs pull：pull=fetch+merge，fetch 只下载不合并（可先 diff 再决定）；merge 保留分叉历史（真实但乱）vs rebase 线性化（干净但改写历史，共享分支禁 rebase）；多人协作流：feature 分支 → PR/MR → review → squash merge；拉远程分支：`git fetch && git checkout -b local origin/branch`。

**G5. torchrun 参数与批量杀进程？**
`--nproc_per_node=N --nnodes=x --node_rank=y --master_addr --master_port`（RDZV 配置）；批量杀：`pkill -f torchrun` 或 `ps aux | grep torchrun | awk '{print $2}' | xargs kill -9`。

**G6. Ray 底层？**
GCS（全局状态/etcd）+ Driver（提交任务图）+ Worker 进程池 + **Raylet**（每节点调度器：任务队列+对象存储）。核心抽象：remote 函数/Actor（有状态）+ Plasma 对象存储（零拷贝共享）。特性：异构资源（GPU/CPU 标签）、容错（ lineage 重放）。

## H. 开放与职业题

**H1. Infra 前沿方向？（准备三分钟版）**
推理侧：KV Cache 池化/分层（Mooncake/LMCache 路线之争）、PD 分离与异构集群、超低精度 FP4/FP8、稀疏注意力落地（DSA）、新硬件互连（NVLink-Network/UEC）。可用本库「生产化边界」的趋势判断收尾——展示你有体系观点而非新闻罗列。

**H2. 算法研究 vs Infra 工程的差异？**
算法：假设驱动、实验迭代、指标（精度/损失）驱动、个人为主；Infra：系统驱动、 profiling 迭代、指标（吞吐/延迟/成本）驱动、协作与长期维护为主。共同点：都要 hypothesis→measure→conclude 的循环——这是可迁移能力。

**H3. 大模型安全包括什么？**
越狱（提示注入/角色扮演绕过）、数据投毒（训练期）、成员推断与训练数据泄露、多模态注入、模型窃取、对齐税（过度拒答）。防御分层：训练期（RLHF 安全对齐）、推理期（输入过滤/输出审核/工具权限最小化）、部署期（网络隔离——本库 Mooncake 六道信任边界是现成案例）。

**H4. 模型压缩技术全景？**
量化（PTQ/QAT，本库量化章）、蒸馏（logits 软标签/特征蒸馏）、剪枝（结构化通道/非结构化权重）、稀疏化（2:4 结构稀疏配硬件加速）、低秩分解（LoRA 的逆用）。组合拳：量化+蒸馏（先蒸馏到小模型再量化）。

**H5. 如何快速进入新技术领域？**
本库自身就是示范答案：官方文档建立框架 → 源码验证关键论断（不信转述）→ 输出（写作/对比表是最好的学习）→ 用真实考题校验理解。附追问「如何验证计算精度/数值稳定性」：单元测试对拍（参考实现 vs 优化实现）、典型值边界（inf/nan/极小值）、逐层误差累积检查、bf16/fp32 混合精度策略。

**H6. 其余快答。** SID 训练流程：按其论文的隐式扩散/自回归结构作答，先说清你知道它是哪类模型再讲训练目标；chart-to-story 数据：图表解析（结构化抽取）→ 语义描述 → 故事生成，三级数据可合成+人工校验；自动驾驶 HPC：感知 BEV+Transformer、规划 DL 化、车载约束（功耗/延迟确定性）→ 与数据中心 GPU 的核心差异是**实时性与功耗预算**；生成式推荐：把 item 当 token、推荐当序列生成（HSTU/SASRec 路线），正负样本：曝光未点击为负（有偏，需 IPS 纠偏）；项目深挖五连（选型→贡献→难点→量化收益→反思）：提前为每个项目准备数字锚点。
