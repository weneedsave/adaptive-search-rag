import jieba
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))

#构建一个知识库检索器
def build_bm25_retriever(chunks: list[str]):
    if not chunks:
        def empty_retriever(question: str, top_k: int = 10) -> list[str]:
            return []
        return empty_retriever

    corpus = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(corpus)

    def retriever(question: str, top_k: int = 10) -> list[str]:
        query_tokens = tokenize(question)
        scores = bm25.get_scores(query_tokens)
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in indexed_scores[:top_k]]
        return [chunks[idx] for idx in top_indices]
    return retriever

