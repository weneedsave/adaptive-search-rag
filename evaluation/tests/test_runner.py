"""run_case 的容错测试。

背景：一轮评测 20-40 分钟、要调上百次 LLM，国内 + 代理下网络抖动是常态。
实测曾因一次 OpenAIConnectionError 废掉整轮。所以：
  - 单题重试
  - 重试耗尽**不抛异常**，记成 grade="error" 让整轮继续
"""
from evaluation.golden import GoldenCase
from evaluation.runner import run_case


class _FlakyApp:
    """模拟不稳定的图：前 fail_times 次调用抛连接错误，之后正常返回。"""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("模拟网络抖动")
        return {**state, "answer": "ok", "contexts": ["片段"],
                "grade": "correct", "retry_count": 0}


def _case():
    return GoldenCase("kb-01", "kb", "什么是RAG？", "检索增强生成")


def test_run_case_succeeds_without_retry():
    app = _FlakyApp(fail_times=0)
    out = run_case(_case(), app, max_attempts=3)
    assert out["answer"] == "ok"
    assert out["grade"] == "correct"
    assert out["branch_actual"] == "generate"
    assert app.calls == 1          # 一次就成，不该多调


def test_run_case_retries_then_succeeds():
    app = _FlakyApp(fail_times=2)
    out = run_case(_case(), app, max_attempts=3)
    assert out["answer"] == "ok"
    assert out["grade"] == "correct"
    assert app.calls == 3          # 第 3 次成功


def test_run_case_records_error_instead_of_raising():
    """重试耗尽必须返回记录而不是抛异常 —— 抛了就废掉整轮。"""
    app = _FlakyApp(fail_times=99)
    out = run_case(_case(), app, max_attempts=3)

    assert out["grade"] == "error"
    assert out["id"] == "kb-01"
    assert out["type"] == "kb"
    assert out["scores"] == {}
    assert out["contexts"] == []
    assert "ConnectionError" in out["error"]
    assert app.calls == 3          # 恰好重试到上限，不多不少
