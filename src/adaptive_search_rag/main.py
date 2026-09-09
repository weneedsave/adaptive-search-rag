import sys
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from adaptive_search_rag.graph import build_graph


def ask(question: str, thread_id: str = "default") -> str:
    """带 checkpointer 的问答，返回 answer"""
    load_dotenv()
    # TODO1：with上下文管理器创建SqliteSaver
    #读取记忆
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        result = app.invoke(
            {"question": question, "route": "", "contexts": [], "answer": ""},
            config={"configurable": {"thread_id": thread_id}},   # thread_id区分会话
        )
    return result["answer"]


def main() -> None:
    """CLI：循环读问题 → ask → 打印答案"""
    # TODO2：Windows防止控制台中文乱码
    sys.stdout.reconfigure(encoding="utf-8")
    # TODO3：循环交互，同一个thread_id维持会话记忆
    print("RAG问答CLI，输入 exit / quit 退出")
    thread_id = "cli"
    while True:
        user_input = input("请输入问题：")
        if user_input in ("exit", "quit"):
            break
        ans = ask(user_input, thread_id=thread_id)
        print(f"回答：{ans}\n")


if __name__ == "__main__":
    main()
