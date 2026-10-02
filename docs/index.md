---
layout: home

hero:
  name: KV Cache 管理体系分析
  text: LMCache · Mooncake · vLLM
  tagline: 一份面向架构师与推理平台团队的深度技术分析，基于本地源码快照逐项验证
  actions:
    - theme: brand
      text: 阅读指南
      link: /guide/
    - theme: alt
      text: 直达总结论
      link: /guide/conclusions

features:
  - icon: ⚖️
    title: 三系统深度对比
    details: Mooncake 是集群级 KV Cache 基础设施，LMCache 是引擎侧缓存管理层，vLLM 是事实接入协议——架构、通路、存储、协议逐层拆解
  - icon: 🔬
    title: 源码级证据
    details: 每个关键论断标注源码文件与行号，覆盖 Transfer Engine、Store、Connector、MP daemon、Coordinator 等核心组件
  - icon: 🧭
    title: 生态全景
    details: NVIDIA Dynamo/KVBM/NIXL、SGLang HiCache、FlexKV、Tair KVCache 等横向组件的能力矩阵与选型建议
  - icon: 🛡️
    title: 工程视角
    details: 可靠性协议、威胁模型、故障路径、生产化边界与风险登记册，区分生产能力与预研特性
---
