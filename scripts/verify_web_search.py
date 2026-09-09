# scripts/verify_web_search.py
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# 先裸调一次，看真实响应结构
r = httpx.post(
    "https://api.bochaai.com/v1/web-search",
    headers={
        "Authorization": f"Bearer {os.getenv('BOCHA_API_KEY')}",
        "Content-Type": "application/json",
    },
    json={"query": "什么是RAG", "count": 2, "summary": True, "freshness": "noLimit"},
    timeout=30.0,
)
print("HTTP 状态码：", r.status_code)
data = r.json()
print("顶层字段：", list(data.keys()))

# 看 webPages 到底在不在顶层
if "webPages" in data:
    value = data["webPages"]["value"]
    print("webPages 在顶层，条数：", len(value))
    for item in value:
        print("  -", item.get("name"), "|", item.get("url"))
elif "data" in data and isinstance(data["data"], dict):
    print("顶层 data 的子字段：", list(data["data"].keys()))
else:
    print("未找到 webPages，data 类型：", type(data["data"]))
