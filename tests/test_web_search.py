# tests/test_web_search.py
from unittest.mock import patch, MagicMock
from adaptive_search_rag.tools.web_search import web_search


def _fake_response():
    """造一个假的 httpx 响应对象"""
    fake = MagicMock()
    fake.json.return_value = {
        "data":{
        "webPages": {                       # ←层级先按官方文档（顶层 webPages）
            "value": [
                {"name": "标题1", "url": "https://example.com/1", "summary": "摘要1"},
                {"name": "标题2", "url": "https://example.com/2", "summary": "摘要2"},
                {"name": "标题3", "url": "https://example.com/3", "summary": "摘要3"},
            ]
        }
    }}
    return fake


def test_web_search_returns_list():
    with patch("adaptive_search_rag.tools.web_search.httpx.post", return_value=_fake_response()):
        results = web_search("测试", top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3


def test_web_search_contains_title_and_url():
    with patch("adaptive_search_rag.tools.web_search.httpx.post", return_value=_fake_response()):
        results = web_search("测试", top_k=3)
    assert "标题1" in results[0]          # 每条含标题
    assert "https://example.com/1" in results[0]   # 含链接
