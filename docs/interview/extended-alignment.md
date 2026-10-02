---
prev: /interview/cuda-systems-answers
next: /interview/extended-rag-agent
---

# 扩展题库 · 对齐与训练（逐题精答）

> 80 题逐题精答，覆盖对齐/RLHF（最高频扩展方向）、SFT 与数据工程、LoRA、RL+MoE。高频题加粗标记；答法保持「结论先行 + 公式/数字锚点」。

## A. PPO 全家桶（高频）

**A1. 写出 PPO 在 RLHF 中优化的目标函数并逐项解释。**
`L = E[min(r_t·A_t, clip(r_t, 1−ε, 1+ε)·A_t)] − β·KL(π‖π_ref)`。`r_t = π(a|s)/π_old(a|s)` 是新旧策略概率比；`A_t` 优势函数（该动作比平均好多少）；clip 把 r 限制在 [1−ε, 1+ε] 防止单步更新过大；KL 项把策略拴在参考模型（SFT 模型）附近防奖励破解（reward hacking）与语言能力崩坏。β 是 KL 系数。

**A2. Clip 在优势为正/负时分别怎么限制？**
不对称：**A>0（好动作）时上限被 clip**——r>1+ε 后继续增大不再增加目标，防过度乐观地加大概率；**A<0（坏动作）时下限被 clip**——但注意下界是 1−ε：若策略已把该动作概率压得很低（r→0），min(r·A, clip·A) 取 r·A（负得更多），**仍允许继续压低**——即「防爆炸不防压缩」，坏动作可以持续压制。

**A3. GAE 怎么算？λ 的作用？**
`A_t = Σ_{l≥0} (γλ)^l · δ_{t+l}`，其中 `δ_t = r_t + γV(s_{t+1}) − V(s_t)` 是 TD 误差。λ→0 退化为单步 TD（偏差大、方差小），λ→1 退化为蒙特卡洛（方差大、无偏差）；λ=0.95 折中。本质是用偏差换方差的指数加权。

**A4. PPO 是 on-policy 还是 off-policy？为什么要重要性采样？**
严格说是 on-policy（样本必须由当前策略采出）；重要性采样 `r_t` 是**复用同一批样本多次 epoch 更新**时的修正——第一次更新后 π 已偏离 π_old，直接用旧样本梯度有偏，比值修正把它拉回来。与 off-policy（如 SAC 用 replay buffer）的区别：PPO 的复用窗口只有一个 batch 内的几个 epoch。

**A5. Trust Region 与 PPO 的关系？**
TRPO 用二阶近似求解「KL 约束下最大化目标」（解线性方程、共轭梯度，实现复杂）；PPO 用一阶 clip 近似同一目标——把「硬约束」换成「软惩罚/截断」，实现简单、可大规模并行。演进：REINFORCE（无约束）→ TRPO（信任域）→ PPO（clip 近似）。

**A6. 优势函数怎么计算？Critic 怎么更新？**
A = GAE 加权的 TD 误差和（A3 公式）；Critic（价值网络）回归 `V(s)` 目标为 λ-return `R_t = Σ(γλ)^l r_{t+l} + ...`，损失 MSE。RLHF 中 Critic 通常从 SFT 模型初始化加价值头。

**A7. RLHF 完整流程涉及哪些模型角色？（×3 问法）**
四个：**Policy**（被训练的对话模型）、**Reference**（冻结的 SFT 模型，KL 锚点）、**Reward Model**（打分）、**Critic**（估值）。GRPO 砍掉 Critic。数据流：policy 采样 → RM 打分 → 算优势 → 更新 policy；每步与 reference 算 KL。

## B. DPO

**B1. DPO 损失函数与训练目标？相较 RLHF 的改进？**
`L = −log σ(β[log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)])`。直觉：直接优化「偏好对中赢家的对数比减输家的对数比」，数学上等价于 RLHF 最优解的闭式重参数化。改进：省去 RM 和采样循环（离线偏好对直接训），训练稳定、成本低。

**B2. PPO 与 DPO 核心区别？（×5 问法）**
PPO 在线采样 + RM 打分（on-policy，探索性强、能发现训练分布外行为）；DPO 离线偏好对（分布受限在数据内）。PPO 效果常更好的原因：偏好数据静态、可能过拟合标注偏差；DPO 对数据质量/分布更敏感。多轮 DPO（iterative DPO）：用当前模型生成、RM/LLM 重标注、再训——部分弥补在线性。

**B3. DPO 训练后输出明显变长怎么应对？（×2）**
长度偏置被学进偏好对（标注者偏爱长回答）。应对：偏好对去长度相关（配对时长度匹配）、损失加长度正则、换 SimPO（在 DPO 基础上显式去偏）、后处理长度惩罚。

## C. GRPO 与变体（DeepSeek 系高频）

**C1. GRPO 原理？与 PPO 核心区别？**
组相对策略优化：对同一 prompt 采样 G 个回答，**用组内奖励的均值归一化当优势**（`A_i = (r_i − mean)/std`），省掉 Critic。区别一句话：PPO 的基线是学出来的价值网络，GRPO 的基线是同组兄弟的平均——**用一个 prompt 的多次尝试互为对照**。省一整个 Critic 模型的显存与训练不稳定性（Critic 与 Policy 学不同步是 PPO 的老大难）。

**C2. 序列级损失怎么分配到 token？序列级平均 vs 批次级平均？**
默认把序列优势均摊到每个 token（所有 token 同权重）；GSPO（Group Sequence Policy Optimization）指出这会被长序列稀释——改为序列级整体权重（先按序列算 logprob 再聚合），长短序列话语权均衡。DAPO：动态采样（过滤全对/全错组——组内无区分度的样本是噪声）+ 过长回答截断不惩罚。

**C3. 从策略梯度到 GRPO 的演进脉络？**
REINFORCE（`∇J = E[A·∇log π]`，保留：策略梯度定理）→ baseline（减均值降方差，保留：GRPO 的组均值就是它）→ Actor-Critic（加 Critic，GRPO 舍弃）→ TRPO/PPO（信任域/clip，保留）→ GRPO（组基线替 Critic）。**保留的是降方差与稳更新，舍弃的是价值网络的滞后性**。

**C4. 奖励函数怎么设计？GRPO 要冷启动吗？**
RLVR（可验证奖励）场景：数学答案对错、代码测试通过、格式合规——规则可判，无需 RM。开放场景仍需 RM 或 LLM-as-judge。冷启动：通常先 SFT 一轮格式/指令遵循（否则初期采样全乱、组内无信号）；DeepSeek R1 的 R1-Zero 证明纯 RL 也可冷启动但需要规则奖励与大量算力。

## D. Reward Model 与评估

**D1. RM 训练流程与目标？**
Bradley-Terry 损失：`L = −log σ(r(x,y_w) − r(x,y_l))`——让赢家分数高于输家，只学相对序不学绝对分。数据：同 prompt 两个回答的人工/AI 偏好标注。Reward Bench 分类：Chat、Chat-Hard、Safety、Reasoning 四类。

**D2. LLM as Judge 怎么做？偏差？**
用强模型按 rubric 打分或成对比较。三种系统性偏差要主动答：**位置偏差**（偏爱第一个选项——交换位置再判一次取平均）、**长度偏差**（偏爱长回答——长度归一）、**自我偏好**（偏爱同家族模型）。

## E. SFT 与数据工程（高频）

**E1. 为什么 SFT 之后还要 RLHF？（×4 问法，三层答案）**
① 目标不同：SFT 模仿正例（学「什么样子」），RLHF 优化偏好序与负反馈（学「什么更好」）；② 覆盖不同：SFT 数据有限，长尾行为靠 RL 探索补齐；③ 对齐目标（有用/无害/诚实）本质是偏好排序，正例模仿表达不了「不要这样做」。

**E2. SFT 流程与数据构建？**
模型从基座初始化；数据 = 指令-回答对（质量 > 数量，LIMA 证明 1K 高质量可对齐；多样性：任务类型/难度/风格分层）；损失只算 assistant 段。框架：DeepSpeed/FSDP + packing 提吞吐。

**E3. 多轮对话 Loss Mask 怎么设计？**
只对 assistant 回复 token 计损失，user 段与 padding 置 −100（忽略）。追问点：多轮中后续轮次的 user 输入依赖前面 assistant 输出，训练时要完整前缀进模型但只监督 assistant 位——**mask 的是损失不是输入**。

**E4. packing 与多轮对话形式的区别？**
packing 把多个独立样本拼进一个序列提吞吐；风险是**跨样本注意力污染**（后面的样本看到前面样本的内容）——需重置 position_ids + 文档级 attention mask 隔离。多轮对话是天然的连续序列，无污染问题但不能随意切分（切点必须在轮次边界）。

**E5. 手写 SFT loss（shift-right 对齐）。**
`logits[:, :-1, :]` 与 `labels[:, 1:]` 对齐：位置 i 的 logits 预测第 i+1 个 token；`loss = F.cross_entropy(logits[:, :-1].flatten(0,1), labels[:, 1:].flatten(), ignore_index=-100)`。为什么 shift：自回归模型在看到 token t 后预测 t+1，监督信号要错开一格。

**E6. SFT 后特定任务增强但通用能力下降？**
灾难性遗忘。应对：混入通用数据 replay（如 9:1 任务:通用）、降学习率、LoRA 限制更新子空间、early stop 按通用 bench 巡检。对齐后过于保守/频繁拒绝：偏好数据中 refusal 过多或 KL 过强——稀释 refusal 对、降 β、加「有帮助性」奖励项。

**E7. 预训练 vs SFT 的损失与数据差异？**
损失同源（next token CE）；差异全在数据：预训练海量网络文本（数 T token、质量过滤+去重）、学习率大衰减到小；SFT 精选指令数据（万~百万级）、小 LR 短 epoch。预训练数据清洗：去重（MinHash/精确）、质量分类器过滤、毒性/PII 处理、多语言配比。

**E8. 多轮超长上下文的训练方案？**
序列并行（ring attention/DeepSpeed-Ulysses）切注意力、数据上截断最旧轮次但保留摘要、渐进式长度扩展（短→长 curriculum）。推理侧配合本库话题：超长多轮是 prefix cache 与 KV 池化的最大受益场景。

**E9. Qwen 微调的阶段性策略与 loss 考虑？**
先全参或 LoRA SFT 指令对齐 → 领域数据继续训（混通用防遗忘）→ 需要偏好时 DPO/PPO。loss 考虑：任务数据权重、mask 策略（是否监督思考过程）、长度均衡。

## F. LoRA（×6 高频）

**F1. 原理与核心思想？**
`W' = W + BA`，B∈R^{d×r}、A∈R^{r×k}，r≪min(d,k)（如 8/16/64）。训练只更新 BA（参数量降百倍），推理可合并回 W（零额外延迟）或保留挂载（多租户热切换）。低秩假设的依据：微调时参数更新量 ΔW 的本征秩很低（论文实测）——「改变行为的自由度远小于参数量」。

**F2. 为什么 A 高斯初始化、B 初始化为零？反过来行吗？**
B=0 保证初始 BA=0，模型行为与原模型完全一致（训练起点无扰动）；A 若也为零则 `∂L/∂B = Aᵀ∇ = 0`、`∂L/∂A = ∇Bᵀ = 0`——**双零死锁，梯度消失训不动**。A 用高斯保证 B 有非零梯度通路；反过来（A=0, B 高斯）数学上对称可行，惯例用前者。

**F3. rank 大小的影响？rank 不当的现象？**
r 过小：容量不足欠拟合（学不会领域知识）；r 过大：趋近全参微调，过拟合 + 显存回升 + 多任务干扰。经验：简单风格迁移 r=8 够；知识注入需 64+；配合 `lora_alpha`（缩放 α/r）。

## G. RL + MoE 与训练杂题

**G1. RL+MoE 下 reward 引导路由退化？（×2，2026 新题）**
现象：模型发现把请求集中到少数专家能拿更高 reward → 负载崩塌 → EPLB 失效 → 训练 OOM。应对：RL 阶段保留负载均衡辅助损失、reward 加路由熵正则、专家容量硬限 + 溢出丢弃、数据侧打散同质 prompt。

**G2. Muon 与 AdamW 为什么不能混用？**
Muon（正交化二阶矩）适合大 batch 高学习率的预训练期快速下降；后训练阶段 batch 小、需要精细稳定——切换会破坏优化轨迹的曲率假设；且两者隐式学习率尺度不同，直接混用等于突然变 LR。

**G3. 知识蒸馏适合预训练阶段吗？**
分情况：小模型预训练时用大模型 logits 软标签蒸馏（TinyBERT/MiniLM 路线）有效——继承类间相似度知识；但通用预训练的目标是覆盖全网分布，蒸馏上限被教师锁死。主流仍用于后训练压缩。

**G4. CoT 训练数据怎么构造？**
长思维链蒸馏（强模型生成推理轨迹 + 过滤错误答案）、 rejection sampling（保留答对的轨迹）、PRM（过程奖励模型）打分筛选、领域合成（数学/代码可验证）。R1 之后重点是「可验证 + 多样性」。

**G5. 异步 RL 的算法适配？损失怎么修正？**
异步（rollout 与训练 overlap）导致策略滞后——必须 importance sampling 修正 stale 样本（r_t 离 1 越远 clip 越狠）；补偿手段：缩短 rollout 队列、提高 KL 系数、或者 off-policy 化（retracing/重要性加权全量修正）。异步调度方案：单机 asyncio（rollout 引擎与 trainer 交替）、多机 fully-decoupled（独立 rollout 集群 + 参数 server 广播）。

**G6. Rollout 阶段优化？**
Rollout 是推理：套用全部推理优化（continuous batching、投机解码、FP8 量化 rollout）；训推权重同步用 NCCL broadcast 或 in-place 融合；vLLM/SGLang 的 sleep mode 让 rollout 引擎在训练期释放显存。

## H. 后训练全景收束题

**H1. 预训练与后训练各自的目标与内容？**
预训练：next-token 于海量语料，学语言与世界知识（月级、万卡）；后训练：SFT（指令与格式）→ RLHF/GRPO（偏好与推理能力）→（可选）持续学习。一句话：预训练给「能力」，后训练给「行为」。

**H2. 大模型主要结构特征？（速答版）**
Decoder-only Transformer 堆叠：RMSNorm(pre) + QKV attention（GQA/MLA）+ RoPE + SwiGLU FFN + 残差；（MoE：FFN 换专家路由）；千亿级参数、上下文 128K+、训练 bf16/fp8。
