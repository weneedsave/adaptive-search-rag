from adaptive_search_rag.retrieval.bm25 import build_bm25_retriever, tokenize

# 小而可控的语料：每个 chunk 有明确的主题词
CHUNKS = [
    "RAG 通过检索外部知识来减少大模型的幻觉问题",
    "LangGraph 用 StateGraph 定义节点和边，支持条件路由",
    "BM25 是一种基于词频的稀疏检索算法，擅长精确词匹配",
    "交叉编码器 CrossEncoder 会对 query 和文档做深度交互打分",
]

def test_tokenize_returns_list_of_str():
    tokens = tokenize("混合检索")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

def test_bm25_exact_term_wins():
    """专有名词只出现在一个 chunk 里时，BM25应把它排到第一"""
    retriever = build_bm25_retriever(CHUNKS)
    result = retriever("CrossEncoder 是什么", top_k=1)
    assert len(result) == 1
    # 只有最后一个 chunk 含 "CrossEncoder" 这个精确词
    assert "CrossEncoder" in result[0]

def test_bm25_respects_top_k():
    retriever = build_bm25_retriever(CHUNKS)
    assert len(retriever("检索", top_k=2)) == 2

def test_bm25_returns_original_chunks_only():
    """返回的必须是原始 chunk 文本，不是分词后的碎片"""
    retriever = build_bm25_retriever(CHUNKS)
    for doc in retriever("检索算法", top_k=3):
        assert doc in CHUNKS

def test_bm25_empty_corpus_does_not_crash():
    retriever = build_bm25_retriever([])
    assert retriever("任意问题", top_k=3) == []
