// 默认主题 + 客户端 Mermaid 渲染（hydration 后与路由切换后）
import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import type { Theme } from 'vitepress'

const renderMermaid = async () => {
  const blocks = document.querySelectorAll<HTMLDivElement>('div.language-mermaid')
  if (!blocks.length) return
  const { default: mermaid } = await import('mermaid')
  mermaid.initialize({ startOnLoad: false, theme: 'default' })
  for (const container of blocks) {
    if (container.dataset.rendered === '1') continue
    const text = (container.querySelector('pre code') as HTMLElement)?.innerText || ''
    if (!text.trim()) continue
    container.dataset.rendered = '1'
    try {
      const { svg } = await mermaid.render('m' + Math.random().toString(36).slice(2, 8), text)
      container.innerHTML = svg
    } catch { /* 渲染失败保留代码块 */ }
  }
}

const Theme: Theme = {
  ...DefaultTheme,
  Layout: DefaultTheme.Layout,
  enhanceApp({ router }) {
    if (typeof window !== 'undefined') {
      window.addEventListener('load', () => setTimeout(renderMermaid, 100))
      router.onAfterRouteChanged = () => setTimeout(renderMermaid, 300)
    }
  },
}

export default Theme
