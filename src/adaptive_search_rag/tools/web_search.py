# src/adaptive_search_rag/tools/web_search.py
import os
import httpx
from dotenv import load_dotenv


def web_search(query: str, top_k: int = 3) -> list[str]:
    """调博查 API，返回规整后的文本片段列表（每段含标题+摘要+链接）"""
    #加载env
    load_dotenv()

    api_key = os.getenv("BOCHA_API_KEY")
    # 发 POST 请求（关键参数见下方"已确认的 API 规格"）
    response = httpx.post(
        "https://api.bochaai.com/v1/web-search",
        headers={
            "Authorization": f"Bearer {api_key}",
            #返回json
            "Content-Type": "application/json",
        },
        json={"query": query, "count": top_k, "summary": True, "freshness": "noLimit"},
        timeout=30.0,
    )
    response.raise_for_status()   # 非 2xx 抛异常，交给上层降级
    data = response.json()

    # TODO 1：从 data 里取出 webPages.value
    pages = data["data"]["webPages"]["value"]

    # TODO 2：遍历 pages，把每条规整成 "[标题](链接)\n摘要" 塞进列表
    results = []
    for page in pages:
        name = page.get("name", "")
        url = page.get("url", "")
        summary = page.get("summary") or page.get("snippet", "")
        results.append(f"[{name}]({url})\n{summary}")
    return results
