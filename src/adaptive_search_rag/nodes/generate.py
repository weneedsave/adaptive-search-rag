from langchain_core.messages import SystemMessage, HumanMessage
from adaptive_search_rag.state import RAGState
from adaptive_search_rag.llm import get_llm

SYSTEM_PROMPT = """你是知识库问答助手。请严格基于【给定资料】回答用户问题。
规则：
1. 只使用资料中提供的信息作答；
2. 资料中没有的内容，明确说"根据现有资料无法回答"，不要编造；
3. 回答要简洁、准确、有逻辑。
"""


def generate_node(state: RAGState, llm=None) -> dict:
    """基于 question + contexts 生成回答，返回 {"answer": ...}"""
    if llm is None:
        llm = get_llm()
    question = state["question"]
    contexts = state.get("contexts", [])   # .get 兜底，防止键缺失

    if not contexts:
        # TODO1：没有参考资料，直接返回固定提示，**不调用LLM**
        return {"answer": "根据现有资料无法回答"}

    # ===== 正常：有上下文 =====
    # TODO2：把 contexts 拼成一段文本，每条参考资料编号分隔
    context_text = "\n\n".join([f"【参考资料{i+1}】{chunk}" for i, chunk in enumerate(contexts)])

    # TODO3：拼接用户消息：问题 + 参考资料
    human_msg_content = f"""参考资料：{context_text}

用户问题：{question}
"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_msg_content)
    ]

    # TODO4：调用LLM，读取 .content
    resp = llm.invoke(messages)
    answer = resp.content

    return {"answer": answer}
