---
prev: /ecosystem/engineering
next: /ecosystem/recommendations
---

本篇把 KV cache 当作多租户状态系统做安全验证：跨租户 key 冲突、cache poisoning、删除语义、数据驻留、加密与审计取证，共六族 30+ 个可执行测试用例。核心立场是——如果这些测试失败，平台不应上线共享池。

> 来源：V8 全文。

# 威胁模型与安全测试方案

本文把 KV cache 当作多租户状态系统做安全验证。目标不是证明系统实现了所有安全能力，而是给出可执行测试：如果结果失败，平台不应上线共享池。

## 1. 资产、信任边界与威胁者

### 1.1 资产

| 资产 | 说明 |
|---|---|
| KV tensor | 虽然不是 prompt 明文，但可服务上下文重建或语义侧信道 |
| key/hash/token metadata | 可能暴露 prefix 结构、会话长度、用户行为 |
| cache directory | 暴露对象位置、租户、模型、版本和生命周期 |
| deletion marker | 决定用户/租户删除是否真正生效 |
| audit log | 取证与合规的核心证据 |
| transport metadata | hostname、segment、block address、rkey、rank |

### 1.2 信任边界

```text
用户/租户
  -> API/Router
    -> vLLM Scheduler/Worker
      -> Cache Connector
        -> Local L1 / P2P peer / L2 store / shared pool
          -> SSD / object store / metadata service
```

必须假设：

1. 恶意租户可提交精心构造的 prompt；  
2. 恶意 worker/connector 可能伪造 lookup/store；  
3. peer node 可能被攻陷；  
4. metadata 与 data plane 可能被分开攻破；  
5. 备份/日志可能留存超过请求生命周期。

### 1.3 威胁者

| 威胁者 | 目标 |
|---|---|
| 租户 A | 读/污染租户 B 的 KV |
| 恶意内部用户 | 让模型复用错误上下文 |
| 被攻陷 worker | 写伪造 block 或扩大配额 |
| 被攻陷 peer | 返回部分/篡改/过期 KV |
| 供应链攻击 | 修改 connector、tokenizer hash、model revision |
| 内部运维 | 绕过租户隔离或删除审计 |

## 2. 统一 key schema 测试矩阵

所有系统都必须回答：两个 key 什么时候允许相等？

至少包含以下字段：

```yaml
key_schema:
  schema_version: 1
  tenant_id: required
  namespace: session|system_prompt|public_model|tenant_shared
  user_id: optional
  session_id: optional
  model_id: required
  model_revision: required
  quantization: required
  tokenizer_revision: required
  adapter_id: optional
  adapter_revision: optional
  attention_type: mha|gqa|mla|mamba|hybrid
  parallel_layout: tp/pp/dp/ep
  block_size: required
  chunk_size: required
  prefix_tree_root: required
  block_hashes: required
  multimodal_hashes: optional
  data_region: required
  cache_salt: required
  object_state: partial|complete
  created_at
  expires_at
```

最小隔离规则：

1. `tenant_id` 不同，即使 token/hash 相同也不可复用；  
2. model/tokenizer/quantization 不同，不可复用；  
3. adapter 不同，除非明确声明 LoRA-safe；  
4. attention type、TP/PP/DP/EP shard 不同，不可直接复用；  
5. partial block 不可当作 complete；  
6. data_region 不同，不可跨 region fetch；  
7. public/system prompt 必须使用独立 namespace，不得与用户 session 混合。

## 3. 跨租户 key 冲突测试

### T-KEY-01：基础跨租户冲突

步骤：

1. 租户 A 和 B 使用完全相同 prompt 生成 KV；  
2. 记录 A 的 key `KA`、B 的 key `KB`；  
3. 删除 A 会话后，用 B 请求相同前缀；  
4. 检查 B 是否从 A 的对象读取。

通过：

1. `KA != KB`；  
2. B 查询不到 A；  
3. 存储层没有跨租户 replica；  
4. audit 显示 B 为 miss。

### T-KEY-02：model/tokenizer/quant 冲突

矩阵：

| 维度 | A | B | 期望 |
|---|---|---|---|
| model revision | r1 | r2 | miss |
| tokenizer | v1 | v2 | miss |
| quantization | fp16 | fp8 | miss |
| attention type | MHA | MLA | miss |
| TP/PP shard | 8/1 | 4/2 | 不直接复用 |
| block size | 16 | 32 | miss |
| LoRA | none | adapter-x | miss |

通过：所有 row 的 KV 字节级隔离，且 metadata 不混淆。

### T-KEY-03：cache salt 缺失/回退

步骤：

1. 正常配置 `cache_salt=tenant_id`；  
2. 启动一个错误配置实例，`cache_salt=""` 或固定默认值；  
3. A 写入敏感 prompt；  
4. B 请求相同 prompt。

通过：

1. 错误实例无法加入共享池，或启动失败；  
2. 缺少 tenant salt 的写被拒绝；  
3. 不允许静默 fallback 到 global namespace。

### T-KEY-04：public/system prompt 边界

步骤：

1. 将一段用户 prompt 放入 `tenant_shared`；  
2. 将相同内容放入 `public_model`；  
3. 逐 token 修改用户尾段；  
4. 查询 public namespace。

通过：

1. public 与 tenant namespace 不互查；  
2. 公共系统提示只能由管理员/admin API 写入；  
3. 用户 prompt 不能提升为 global。

### T-KEY-05：partial tail 与 complete 混淆

步骤：

1. 写入一个只有 7.5 个 block 的长前缀；  
2. 构造最后一个 partial block；  
3. 触发 CoW/flush/restart；  
4. 用另一个会话查询 7.5 block 和 8 block。

通过：

1. partial 不作为 complete 发布；  
2. 7.5 block 不能命中 8 block；  
3. CoW 后旧 partial 被正确清理；  
4. group/object completeness 可审计。

## 4. Cache poisoning 测试

### T-POISON-01：同 key 直接覆盖

步骤：

1. 租户 A 写入正确 KV；  
2. A 用 API/connector 重放同 key，但写入不同 token/KV；  
3. B 读取。

通过：

1. immutable object 不可覆盖；  
2. Upsert 需要授权和版本条件；  
3. 覆盖尝试进入 audit/security event。

### T-POISON-02：partial write 抢占发布

步骤：

1. 发起 PutStart；  
2. 只写部分 replica/slice；  
3. 立即读；  
4. 不发 PutEnd，等待 zombie cleanup。

通过：

1. PutEnd 前不可见；  
2. 超时后空间释放；  
3. 不留下可读 partial；  
4. 后续同 key PutStart 可安全抢占。

### T-POISON-03：bit-flip / storage 篡改

步骤：

1. 写入 KV 并计算 reference hash/CRC；  
2. 在 SSD/object 层修改若干字节；  
3. 清空客户端热缓存；  
4. 读取目标对象。

通过：

1. CRC/hash/checksum 失败；  
2. 返回 miss/error，不返回篡改 KV；  
3. 产生 integrity alert 和 quarantine。

### T-POISON-04：hash collision

步骤：

1. 找到两个 prefix 使 block hash 相同但 token id 不同；若成本高，用受控 fake hasher 注入；  
2. A 写第一个，B 查询第二个。

通过：

1. 命中前进行 token id / Merkle root 验证；  
2. 不只依赖短 hash；  
3. 验证失败不返回 KV 并报警。

### T-POISON-05：恶意 connector/worker

步骤：

1. worker 直接调用底层存储 API 写正确 key + 错误 value；  
2. worker 伪造 `get_num_new_matched_tokens()` 返回外部命中；  
3. worker 提前返回 store 完成。

通过：

1. connector 使用 mTLS/身份；  
2. 写权限最小化，store job 需要 scheduler 发起的 generation/fence；  
3. store 完成必须等所有 rank；  
4. 早报完成不释放 GPU block 引用；  
5. 异常写入进入 quarantine。

### T-POISON-06：P2P peer 篡改

步骤：

1. 在 peer 返回的 address 中指向越界/错误对象；  
2. peer 在 RDMA read 中替换内容；  
3. peer 返回过期 lock/address。

通过：

1. address 必须落在锁定 object 范围；  
2. transfer 有长度/权限/校验；  
3. lock 与 incarnation 绑定；  
4. peer 重启后旧 address 失效。

### T-POISON-07：压缩/非前缀复用污染

针对 CacheGen/CacheBlend：

1. 压缩对象解码后必须做模型可接受性校验或灰度标签；  
2. 非前缀 fragment 不得伪装成完整 prefix；  
3. 质量恢复失败必须回退 recompute；  
4. research 特性默认关闭，且带 schema/maturity 标签。

## 5. 删除语义测试

### T-DEL-01：用户删除会话

步骤：

1. 租户 A 创建 session 并写 KV；  
2. 调用删除会话 API；  
3. 立即和延迟 1s/10s/60s 查询相同 prefix；  
4. 检查 SSD、replica、日志、备份指针。

通过：

1. 读立即 miss；  
2. 所有 replica 在 SLA 内删除；  
3. SSD/对象存储无可读 value；  
4. audit 记录请求者、删除原因、key hash、对象数量。

### T-DEL-02：租户 purge

步骤：

1. 租户 A 写 10k objects，包括 hot/pinned/SSD/replica；  
2. 执行 tenant purge；  
3. 并发读写 A/B。

通过：

1. A 全部不可读；  
2. B 不受影响；  
3. pinned object 根据权限强制删除或明确拒绝；  
4. quota 释放；  
5. 未删除对象数为 0。

### T-DEL-03：TTL/LRU 与 active read 竞争

步骤：

1. 对象 TTL 设置 1s；  
2. read 持续 3s；  
3. 并发触发 eviction。

通过：

1. lease/lock 保护中的对象不删除；  
2. 过期后读失败而不返回脏数据；  
3. 不出现删除后重读旧 value。

### T-DEL-04：异步删除崩溃

步骤：

1. 发起删除；  
2. 在后台删除线程执行时 kill metadata/master/client；  
3. 恢复服务后查询。

通过：

1. deletion marker 持久化或在恢复后重放；  
2. 不留下可读 canary；  
3. 不出现 metadata 已删但 value 可通过旧 replica list 读到。

### T-DEL-05：snapshot/OpLog 恢复后 resurrection

步骤：

1. Put key -> snapshot；  
2. Delete key -> 新 OpLog；  
3. crash -> restore snapshot + replay；  
4. 查询 key。

通过：

1. delete OpLog 正确应用；  
2. key 不复活；  
3. 备份可标记为不可服务或需安全恢复审批；  
4. soft pin 状态降级不会导致删除对象复活。

### T-DEL-06：GPU block 与 external object 不一致

步骤：

1. vLLM 正在异步 store；  
2. 触发请求删除/abort/CoW；  
3. store job 后台继续 DMA；  
4. 删除后立即分配新 request 复用 block。

通过：

1. store job 引用 block 直到完成；  
2. 删除外部 key 不写回本地；  
3. 新 request 读不到旧 KV；  
4. store job 不复活已删除 key。

## 6. 数据驻留与跨 Region 测试

### T-RES-01：region pinning

步骤：

1. 租户 A 要求 region=cn-east；  
2. 写入 KV；  
3. 查询 object placement、replica、P2P peer、SSD、backup。

通过：

1. 所有 data plane 对象只在允许 region；  
2. metadata 中的 region policy 被强制执行；  
3. 跨 region peer lookup 被拒绝。

### T-RES-02：P2P 跨 region 泄漏

步骤：

1. region A 与 region B 网络可达；  
2. A 请求 prefix；  
3. 观察 B 是否被选为 peer。

通过：

1. peer discovery 按 region 过滤；  
2. 除非 policy 显式允许，跨 region RDMA 不发生；  
3. audit 记录跨 region 尝试。

### T-RES-03：metadata 与 data plane 分离

数据不跨 region 不等于合规安全。检查：

| 对象 | 是否可跨 region |
|---|---|
| key hash | 默认不允许，除非策略允许 |
| block hash/token metadata | 默认不允许 |
| object size | 策略决定 |
| access log | 策略决定 |
| audit log | 建议保留在租户 region 或加密集中区 |

## 7. 加密、密钥与完整性

### 7.1 最低要求

| 层 | 要求 |
|---|---|
| transport | mTLS 或等价身份认证；RDMA fabric 至少有隔离/加密策略 |
| metadata service | TLS + authz |
| object at rest | AES-GCM 等认证加密，或依赖磁盘加密并有明确边界 |
| key management | per-tenant KEK，KMS envelope encryption，支持 revoke/rekey |
| integrity | header/key/value/offset 覆盖 CRC 或 MAC |
| logs | 不写 prompt 明文，key hash 可审计但不可反推明文 |

### 7.2 测试

#### T-ENC-01：落盘 canary

1. 写入租户 canary KV；  
2. 直接读取 SSD/object 文件；  
3. 搜索 token 序列、block hash 明文和 canary pattern。

通过：文件层无法读出明文 KV，或系统明确证明磁盘卷加密为唯一信任边界。

#### T-ENC-02：per-tenant key isolation

1. A/B 使用不同 KEK；  
2. 伪造 A 的 DEK 去解 B；  
3. revoke A 的 KEK。

通过：跨 KEK 解密失败；revoke 后 A 立即 miss，不会返回明文。

#### T-ENC-03：transport downgrade

1. 关闭 TLS/cert verification；  
2. 中间人拦截 metadata RPC；  
3. 修改 block address/size。

通过：系统拒绝非信任连接；不静默 fallback；拦截产生安全事件。

#### T-ENC-04：完整性覆盖范围

构造修改：

1. 只改 value byte；  
2. 只改 key；  
3. 只改 offset；  
4. 只改 length；  
5. 只改 object state。

通过：任何字段篡改都会导致校验失败或签名/授权失败。

## 8. 审计与取证

### 8.1 最小事件

| 事件 | 必须包含 |
|---|---|
| lookup hit/miss | tenant, model, key hash, schema, result, request id |
| store start/end | tenant, job id, object keys, rank, bytes, result |
| read | tenant, key hash, replica id, lease id, result |
| delete | actor, tenant, policy, key/object count, result |
| eviction | object key, tenant, reason, replacement object |
| integrity failure | object id, expected/actual CRC type, quarantine id |
| authz failure | subject, resource, action, source |
| cross-region attempt | source region, target region, action, verdict |

禁止在 audit 中写入 prompt 明文、完整用户内容、未加密 KV。

### 8.2 测试

#### T-AUD-01：不可抵赖链

1. A 写、读、删除；  
2. B 尝试越权读；  
3. 注入 integrity failure。

通过：每个事件都有 actor、source、time、request/job id、结果和策略版本。

#### T-AUD-02：日志删除一致性

1. 删除对象；  
2. 检查 audit 是否仍保留删除事件；  
3. 确认 audit 不保留可读 value。

通过：删除事实可审计，数据内容不可恢复。

#### T-AUD-03：日志完整性

1. 修改本地日志；  
2. 重放旧日志；  
3. 删除一条安全事件。

通过：日志有 append-only/hash chain/外部 sink；篡改可检测。

## 9. 测试平台设计

### 9.1 组件

```text
Threat Runner
  -> Tenant A/B Victim Requests
  -> Attacker Client/Worker/Peer
  -> Storage Canary Inspector
  -> Metadata/Directory Probe
  -> Network MITM/Fault Injector
  -> Audit Collector
  -> Policy Verifier
```

### 9.2 canary 设计

| canary | 用途 |
|---|---|
| unique token sequence | 判断是否命中他人 KV |
| unique block hash | 判断 key 冲突 |
| encrypted value marker | 判断落盘泄漏 |
| deletion marker object | 判断删除残留 |
| replica id | 判断跨 region/peer 泄漏 |
| poisoned but invalid KV | 判断系统是否做语义校验 |

### 9.3 自动断言

```text
PASS if:
  cross_tenant_hit == 0
  poison_served == 0
  partial_served_as_complete == 0
  undeleted_canaries == 0
  cross_region_objects == 0
  integrity_alerts >= injected_faults
  audit_coverage == 100%
  audit_contains_prompt_plaintext == false
```

## 10. 按系统集成要点

### 10.1 LMCache

重点测：

1. MP connector 的 cache salt 是否进入所有 key；  
2. MP daemon 重启后 old key/IPC handle 是否被隔离；  
3. P2P peer 重启后 lock/address 是否失效；  
4. Redis/Valkey/S3 后端的删除是否覆盖对象与 index；  
5. Mooncake L2 adapter 是否继承 tenant policy。

### 10.2 Mooncake

重点测：

1. tenant id 是否进入 object key 与 group；  
2. PutStart/PutEnd 能否阻止 partial read；  
3. lease expiry 是否只造成 miss；  
4. hard pin 在租户 purge 时的授权边界；  
5. snapshot/OpLog 恢复是否复活已删除对象；  
6. SSD CRC/restart 后 stale value 是否 quarantine；  
7. cross-region peer/master/backup placement。

### 10.3 vLLM

重点测：

1. request id 与 KV key 不产生跨租户 alias；  
2. MultiConnector 任一 connector 不可绕过 policy；  
3. preemption 后 producer connector 不保存已释放 block；  
4. KV event 不把 A worker 的 block 误报为 B 可读公共前缀；  
5. router 不得基于未隔离 cache directory 做跨租户路由。

## 11. 上线判定

以下任何一项失败，共享 KV cache 不应进入生产：

| 级别 | 失败条件 |
|---|---|
| P0 | 跨租户 hit；partial 当 complete；删除后可读；篡改后可服务 |
| P0 | audit 缺失越权读写/删除/完整性失败 |
| P1 | 跨 region policy 可被 peer/backup 绕过 |
| P1 | per-tenant encryption/revoke 无效 |
| P1 | snapshot 恢复复活删除对象 |
| P1 | connector 伪造 store 完成导致 block 数据污染 |

允许进入生产的最小状态：

```text
tenant isolation: mandatory
object integrity: mandatory
deletion semantics: mandatory
audit: mandatory
encryption: mandatory per compliance
cross-region control: mandatory if multi-region
global cache sharing: opt-in
research features: disabled by default
```

## 附：LMCache 静态加密的 serde 实现（官方博客 2026-08，增量）

威胁模型定位清晰：**只保护 L2 at-rest（S3/fs/RESP），L0/L1 仍明文，不防有 server 进程权限的内部人**。

实现是 serde（与 fp8/turboquant 量化器同一可插拔变换层，L1↔L2 往返两侧，开箱适配所有 L2 adapter），跑在存储延迟后面不伤 TTFT。密码学细节：

- AES-128-GCM（AEAD 同时拿到机密性+完整性），格式 `[1B version][12B IV][ciphertext||16B tag]`，每 chunk 固定开销仅 29 字节；
- **tag 校验失败变成 load miss**——损坏密文不可能被静默反序列化成 attention 状态（这是第 7 章加密测试的一个现成答案）；
- 密钥模型：cache_salt 选 key（永不是 key 本身），HKDF-SHA256 从 master key 派生——官方诚实承认这是「舰队级 vs 外部」防护而非租户间隔离（后者需 per-tenant KMS + 租户到节点绑定）；
- 已知缺口：元数据（salt、chunk_hash）在对象名中仍可见；密钥轮换需手动（新 master key + 缓存失效重填）；
- 提醒：**压缩不是加密**（KV 高熵压不动），要省 L2 空间应先量化再加密。

## 附二：Mooncake 六道信任边界的现状盘点（一研「安全实践」篇）

默认信任内网假设下，六道边界的现状与最危险的攻击链：

1. **网络边界**：Master RPC `0.0.0.0:50051` 无认证无 TLS；P2P Handshake 接受任意连接；etcd 明文 HTTP；
2. **内存边界（最核心风险）**：RDMA 四要素（addr/length/**rkey**/QP 号）齐全即可 DMA 读写远程内存，远程 CPU 无感。**RKey/LKey 明文存于 etcd/HTTP 元数据** → 完整攻击链：扫 etcd → 提取 rkey → 构造 `IBV_WR_RDMA_READ` → 读 KV Cache/权重/任意进程内存。共享内存 `shm_open` 权限 0666 应改 0600；
3. **数据边界**：SSD 无加密（LUKS 3-5% 性能代价）；RDMA/TCP 明文；已有 NVMe-oF digest 与 OpLog xxHash 校验（防篡改非加密）；
4. **进程边界**：Docker 默认 root；SPDK 需 root/VFIO；无 seccomp/capability dropping；
5. **凭证边界**：AWS/Redis 密钥环境变量明文（/proc/pid/environ 可读）；无硬编码凭证；
6. **编译边界**：全局仅 -fPIC；**唯一启用安全编译的是 Ascend 模块**（stack-protector + RELRO + noexecstack）。

P0 三件事（堵住 90% 攻击面）：RDMA 网络物理隔离 + Master RPC 绑内网 IP + etcd 启用认证。这与本篇第 7 章加密测试的结论互证：LMCache 的 AES-GCM serde 保护 L2 at-rest，Mooncake 的信任模型同样假设内网可信——两者的加密盲区都在 L0/L1 与传输路径。
