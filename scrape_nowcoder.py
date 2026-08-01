"""
牛客网专栏文章抓取脚本
使用 Playwright 登录并抓取专栏下所有文章内容
"""

import asyncio
import os
import re
from pathlib import Path
from playwright.async_api import async_playwright

# ============ 配置 ============
PHONE = "18844543140"
PASSWORD = "2351312lsd"
COLUMN_URL = "https://www.nowcoder.com/creation/manager/columnDetail/0ox51k"
OUTPUT_DIR = r"C:\Users\zhaoxi\Downloads\Agent开发面经"
MERGED_FILE = os.path.join(OUTPUT_DIR, "AI-Agent面试实战_全部文章.md")
# ============ 配置结束 ============


def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = name.strip('. ')
    return name[:80]


async def login(page):
    """登录牛客网"""
    print("[1/4] 正在登录牛客网...")
    await page.goto("https://www.nowcoder.com/login", wait_until="networkidle")
    await page.wait_for_timeout(2000)

    # 切换到手机号登录
    try:
        phone_tab = page.locator('text=手机号登录')
        if await phone_tab.count() > 0:
            await phone_tab.first.click()
            await page.wait_for_timeout(1000)
    except Exception:
        pass

    # 输入手机号
    phone_input = page.locator('input[placeholder*="手机"], input[type="tel"], input[name*="phone"]')
    if await phone_input.count() == 0:
        phone_input = page.locator('input').nth(0)
    await phone_input.first.fill(PHONE)
    await page.wait_for_timeout(500)

    # 输入密码
    password_input = page.locator('input[type="password"]')
    await password_input.first.fill(PASSWORD)
    await page.wait_for_timeout(500)

    # 点击登录按钮
    login_btn = page.locator('button:has-text("登录"), button:has-text("登 录")')
    await login_btn.first.click()
    await page.wait_for_timeout(3000)

    # 验证是否登录成功
    if "login" in page.url.lower():
        print("  注意: 登录可能需要验证码，请在弹出的浏览器中手动完成验证...")
        await page.screenshot(path=os.path.join(OUTPUT_DIR, "login_debug.png"))
        for i in range(30):
            await page.wait_for_timeout(2000)
            if "login" not in page.url.lower():
                print("  登录成功!")
                return True
        print("  登录超时")
        return False

    print("  登录成功!")
    return True


async def get_article_list(page):
    """获取专栏下所有文章链接"""
    print("[2/4] 正在获取专栏文章列表...")
    await page.goto(COLUMN_URL, wait_until="networkidle")
    await page.wait_for_timeout(3000)

    # 滚动加载所有文章
    prev_count = 0
    for _ in range(20):
        articles = page.locator('a[href*="/creation/manager/contentDetail"]')
        current_count = await articles.count()
        if current_count == prev_count and current_count > 0:
            break
        prev_count = current_count
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1500)

    # 收集文章链接和标题
    article_items = await page.locator('a[href*="/creation/manager/contentDetail"]').all()
    articles = []
    seen_urls = set()

    for item in article_items:
        href = await item.get_attribute("href")
        title = await item.inner_text()
        title = title.strip()
        if href and href not in seen_urls and title:
            seen_urls.add(href)
            if href.startswith("/"):
                href = "https://www.nowcoder.com" + href
            articles.append({"title": title, "url": href})

    # 备用: 尝试其他选择器
    if not articles:
        print("  尝试备用选择器...")
        await page.screenshot(path=os.path.join(OUTPUT_DIR, "column_debug.png"))
        all_links = await page.locator('a').all()
        for link in all_links:
            href = await link.get_attribute("href") or ""
            text = await link.inner_text()
            text = text.strip()
            if ("contentDetail" in href or "article" in href.lower()) and text and len(text) > 3:
                if href.startswith("/"):
                    href = "https://www.nowcoder.com" + href
                if href not in seen_urls:
                    seen_urls.add(href)
                    articles.append({"title": text, "url": href})

    print(f"  共找到 {len(articles)} 篇文章")
    return articles


async def scrape_article(page, article, index, total):
    """抓取单篇文章内容"""
    title = article["title"]
    url = article["url"]
    print(f"  [{index+1}/{total}] 正在抓取: {title}")

    try:
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # 获取文章标题
        title_selectors = [
            '.article-title', '.content-title', 'h1', 'h2.title',
            '.post-title', '.detail-title', '[class*="title"]'
        ]
        article_title = title
        for sel in title_selectors:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                t = (await loc.inner_text()).strip()
                if t:
                    article_title = t
                    break

        # 获取文章正文
        content_selectors = [
            '.article-content', '.content-body', '.post-content',
            '.detail-content', '.markdown-body', '.article-body',
            '[class*="content"]', '[class*="body"]', 'article',
            '.ql-editor', '.nc-light-editor'
        ]
        content = ""
        for sel in content_selectors:
            loc = page.locator(sel).first
            if await loc.count() > 0:
                c = (await loc.inner_text()).strip()
                if len(c) > 50:
                    content = c
                    break

        # 备用: 获取整个页面文本
        if not content or len(content) < 50:
            content = await page.evaluate("""() => {
                const body = document.querySelector('body');
                if (!body) return '';
                const clone = body.cloneNode(true);
                clone.querySelectorAll('script, style, nav, header, footer').forEach(el => el.remove());
                return clone.innerText;
            }""")
            content = content.strip() if content else ""

        return {"title": article_title, "content": content, "url": url}

    except Exception as e:
        print(f"  抓取失败: {e}")
        return {"title": title, "content": f"抓取失败: {e}", "url": url}


async def save_articles(articles_data):
    """保存文章到文件"""
    print("[3/4] 正在保存文章...")

    saved_files = []
    for i, article in enumerate(articles_data):
        filename = f"{i+1:02d}_{sanitize_filename(article['title'])}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {article['title']}\n\n")
            f.write(f"> 来源: {article['url']}\n\n")
            f.write(article["content"])

        saved_files.append(filepath)
        print(f"  已保存: {filename}")

    return saved_files


async def merge_articles(articles_data):
    """合并所有文章到一个文件"""
    print("[4/4] 正在合并所有文章...")

    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        f.write("# AI-Agent面试实战专栏 - 全部文章\n\n")
        f.write(f"> 来源: {COLUMN_URL}\n")
        f.write(f"> 共 {len(articles_data)} 篇文章\n\n")
        f.write("---\n\n")

        f.write("## 目录\n\n")
        for i, article in enumerate(articles_data):
            f.write(f"{i+1}. [{article['title']}](#{sanitize_filename(article['title']).replace(' ', '-')})\n")
        f.write("\n---\n\n")

        for i, article in enumerate(articles_data):
            f.write(f"## {i+1}. {article['title']}\n\n")
            f.write(f"> 来源: {article['url']}\n\n")
            f.write(article["content"])
            f.write("\n\n---\n\n")

    print(f"  已合并到: {MERGED_FILE}")


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            success = await login(page)
            if not success:
                print("登录失败，脚本退出")
                return

            articles = await get_article_list(page)
            if not articles:
                print("未找到文章，脚本退出")
                await page.screenshot(path=os.path.join(OUTPUT_DIR, "no_articles_debug.png"))
                return

            articles_data = []
            for i, article in enumerate(articles):
                data = await scrape_article(page, article, i, len(articles))
                articles_data.append(data)

            await save_articles(articles_data)
            await merge_articles(articles_data)

            print(f"\n全部完成!")
            print(f"  单篇文章保存在: {OUTPUT_DIR}")
            print(f"  合并文件: {MERGED_FILE}")

        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
