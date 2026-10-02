import { defineConfig } from 'vitepress'

// KV Cache 管理体系技术分析 · 文档站点配置
// 三层结构：基础原理 / 核心对比（LMCache·Mooncake·vLLM）/ 横向生态与工程实践

const core = [
  {
    text: '开篇',
    items: [
      { text: '阅读指南', link: '/guide/' },
      { text: '总结论', link: '/guide/conclusions' },
    ],
  },
  {
    text: '第一部分 · 基础原理',
    items: [
      { text: 'KV Cache 本质与显存占用', link: '/basics/kv-cache-fundamentals' },
      { text: 'PagedAttention 与注意力形态', link: '/basics/paged-attention' },
      { text: 'Prefix Cache 原理', link: '/basics/prefix-caching' },
      { text: 'KV Cache 压缩与量化', link: '/basics/compression' },
    ],
  },
  {
    text: '第二部分 · 三大系统对比',
    items: [
      { text: '总体架构', link: '/core/architecture' },
      { text: '物理通路：PCIe·NVLink·RDMA·HCCS', link: '/core/transfer-paths' },
      { text: 'Host-GPU-RDMA 寻址与零拷贝', link: '/core/zero-copy' },
      { text: '存储引擎中的数据变形', link: '/core/storage-transforms' },
      { text: 'Prefix Cache 命中与数据搬运', link: '/core/prefix-hit' },
      { text: 'H2D/D2H 数据搬运对比', link: '/core/h2d-d2h-deepdive' },
      { text: '异步保存与块保护对比', link: '/core/async-protection' },
      { text: '准入控制与防挤占对比', link: '/core/admission-control' },
      { text: 'P2P 与 PD 分离', link: '/core/p2p-pd' },
      { text: 'vLLM Connector 对接', link: '/core/vllm-connector' },
      { text: '可靠性与数据库视角', link: '/core/reliability' },
      { text: '控制面与数据面', link: '/core/control-data-plane' },
      { text: 'LMCache 内部机制深潜', link: '/core/lmcache-internals' },
      { text: 'Mooncake 内部机制深潜', link: '/core/mooncake-internals' },
    ],
  },
  {
    text: '第三部分 · 生态与工程',
    items: [
      { text: '生产化边界与公开声明审计', link: '/ecosystem/production-claims' },
      { text: '横向组件对比', link: '/ecosystem/components' },
      { text: 'SGLang 与 vLLM-Ascend 池化设计', link: '/ecosystem/sglang-ascend-pooling' },
      { text: 'MemFabric Hybrid 深度分析', link: '/ecosystem/memfabric' },
      { text: '源码审计要点', link: '/ecosystem/source-audit' },
      { text: '工程难点与设计动机', link: '/ecosystem/engineering' },
      { text: '威胁模型与安全', link: '/ecosystem/security' },
      { text: '风险与架构改造建议', link: '/ecosystem/recommendations' },
      { text: '选型建议', link: '/ecosystem/selection' },
    ],
  },
  {
    text: '第四部分 · 面试题库',
    items: [
      { text: '总览与备考策略', link: '/interview/' },
      { text: '推理系统高频题精答', link: '/interview/inference-answers' },
      { text: 'CUDA 并行与系统题精答', link: '/interview/cuda-systems-answers' },
      { text: '扩展方向 · 导航', link: '/interview/extended-answers' },
      { text: '扩展：对齐与训练精答', link: '/interview/extended-alignment' },
      { text: '扩展：RAG 与 Agent 精答', link: '/interview/extended-rag-agent' },
      { text: '扩展：ML/CV 与工程基础精答', link: '/interview/extended-engineering' },
    ],
  },
  {
    text: '附录',
    items: [
      { text: '验证方法与指标', link: '/appendix/benchmark' },
      { text: '证据索引与版本', link: '/appendix/evidence' },
      { text: '术语表', link: '/appendix/glossary' },
    ],
  },
]

// GitHub Pages 部署时由 CI 注入 BASE_PATH=/repo-name/，本地预览保持 '/'
const base = process.env.BASE_PATH || '/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: 'KV Cache 管理体系分析',
  description:
    'LMCache、Mooncake、vLLM 及周边生态的 KV Cache 管理体系深度技术分析',
  srcDir: '.',
  outDir: '../dist-docs',
  cleanUrls: true,
  ignoreDeadLinks: true,
  markdown: {
    lineNumbers: false,
    mermaid: true,
  },
  themeConfig: {
    outline: { level: [2, 3] },
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
    sidebar: {
      '/guide/': core,
      '/basics/': core,
      '/core/': core,
      '/ecosystem/': core,
      '/interview/': core,
      '/appendix/': core,
      '/': core,
    },
    nav: [
      { text: '阅读指南', link: '/guide/' },
      { text: '基础原理', link: '/basics/kv-cache-fundamentals' },
      { text: '系统对比', link: '/core/architecture' },
      { text: '生态与工程', link: '/ecosystem/production-claims' },
      { text: '面试题库', link: '/interview/' },
      { text: '附录', link: '/appendix/evidence' },
    ],
    socialLinks: [],
    footer: {
      message: '基于 2026-08-28/29 本地源码快照与公开资料的分析文档',
      copyright: '内部技术分析材料',
    },
    docFooter: { prev: '上一篇', next: '下一篇' },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '浅色',
    darkModeSwitchTitle: '深色',
  },
})
