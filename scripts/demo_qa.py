"""端到端问答演示：真实跑完整图（路由 → 检索 → 生成）。

用法：.venv/Scripts/python.exe scripts/demo_qa.py
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.main import ask

QUESTIONS = [
    "StateGraph 是什么",                    # 走本地检索，精确术语
    "checkpoint 和 store 有什么区别",       # 走本地检索，两个并列术语
]


def main() -> None:
    for q in QUESTIONS:
        t0 = time.time()
        ans = ask(q, thread_id="demo")
        print("=" * 72)
        print(f"问：{q}   （{time.time()-t0:.1f}s）")
        print("-" * 72)
        print(ans)
        print()


if __name__ == "__main__":
    main()
