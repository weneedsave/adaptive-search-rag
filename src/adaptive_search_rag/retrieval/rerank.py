"""精排（rerank）：粗筛候选池 → 用交叉编码器或 LLM 重排
为什么需要精排：
- 粗筛（向量/BM25）是 Bi-Encoder，query 和 doc 各自独立编码，彼此没有交互
- Cross-Encoder 把 [query, doc] 拼成一对送进模型，能做深度交互，
  准确度高约一个量级，但速度慢约 100 倍 → 所以只对 top-8 候选池做
"""
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from adaptive_search_rag.llm import get_llm

class RelevanceScore(TypedDict):
    """LLM 打分器期望的结构化输出"""
    score: float

SCORE_PROMPT = """你是一个相关性评分器。给定用户问题和一段文档，输出 0 到 1 的相关性分数。
- 1.0 表示文档直接回答了问题
- 0.0 表示完全不相关
只输出分数，不要解释。"""

class CrossEncoderReranker:
    """默认精排器：BAAI/bge-reranker-v2-m3"""
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model=None):
        self.model_name = model_name
        self.model = model          # 注入口：测试传假 model

    def _get_model(self):
        #   局部import，避免顶层加载拖慢测试
        if self.model is None:
            from sentence_transformers import CrossEncoder
            #强制本地加载，避免网络问题
            self.model = CrossEncoder(self.model_name, local_files_only=True)
        return self.model

    def rerank(self, question: str, docs: list[str], top_k: int) -> list[str]:
        #问题,相关文档,保留个数
        if not docs:
            return []
        pairs = [[question, d] for d in docs]
        scores = self._get_model().predict(pairs)#交给交叉编码器做深度交互
        scored_pairs = sorted(
            zip(docs, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )
        return [doc for doc, score in scored_pairs[:top_k]]



class NoOpReranker:
    """降级实现：CrossEncoder 加载失败时使用，不做精排，按原顺序截断"""
    def rerank(self, question: str, docs: list[str], top_k: int) -> list[str]:
        # TODO9：直接切片截取
        return docs[:top_k]


class LLMReranker:
    def __init__(self, llm=None):
        # 只保存原始llm，暂时不做结构化包装
        self.llm = llm

    def rerank(self, question: str, docs: list[str], top_k: int) -> list[str]:
        # 结构化包装【只对自建的 llm 做】，注入进来的原样使用
        if self.llm is None:
            llm = get_llm().with_structured_output(
                RelevanceScore,
                method="function_calling"
            )
        else:
            llm = self.llm

        scored_docs = []
        for doc in docs:
            messages = [
                SystemMessage(content=SCORE_PROMPT),
                HumanMessage(content=f"问题：{question}\n\n文档：{doc}")
            ]
            try:
                res = llm.invoke(messages)
                score = float(res["score"])
            except Exception:
                score = 0.0
            scored_docs.append((doc, score))

        # 按分数降序排序，截取top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]
