from collections import defaultdict#特殊字典key不存在会建一个而不是报错

VECTOR_K = 10       # 向量粗筛召回数：粗筛要宽，避免漏
BM25_K = 10         # BM25粗筛召回数：与向量对齐
CANDIDATE_K = 8     # 融合后候选池大小：太小漏答案，太大 rerank 慢
RRF_K = 60          # RRF 公式里的 k，业界惯例

def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """多路排名融合，返回降序排列的文档列表
    rankings 里每一路都是「已按相关性降序排好的文档列表」，例如：
        [["docA", "docB"], ["docB", "docC"]]
    返回去重后的文档列表，按 RRF 分数降序。
    关键语义：同一文档出现在多路里时，分数是【多路贡献之和】——
    这正是 RRF 奖励「多路共识」的机制，不是bug。
    """
    #创建一个字典，key是文档，value是分数,float
    score_map = defaultdict(float)
    # TODO2 遍历每一路结果，rank从1开始
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            score_map[doc] += 1.0 / (k + rank)# RRF 公式计算得分
            #("文档B", 0.0326), ("文档A", 0.0321), ("文档C", 0.0318)
    # TODO3 按分数降序排序，返回文档列表
    #传入,要排序的列表,按哪个键排序,降序
    #这里按照第一个元素排序,既是按分数降序
    sorted_docs = sorted(score_map.items(), key=lambda x: x[1], reverse=True)#
    return [doc for doc, score in sorted_docs]# 返回文档列表,只返回文档


def build_hybrid_retriever(vector_retriever, bm25_retriever, reranker):
    """三组件编排
    三个都是必填参数，由调用方（retrieve.py 的 build_retriever）组装传入——
    hybrid只做编排，不负责造真实组件（它不知道 Chroma配置，也不知道模型名）。
    vector_retriever / bm25_retriever : (question, top_k) -> list[str]
    reranker                          : .rerank(question, docs, top_k) -> list[str]
    """
    def retriever(question: str, top_k: int = 3) -> list[str]:
        # TODO1：两路粗召回
        rankings = [
            vector_retriever(question, VECTOR_K),
            bm25_retriever(question, BM25_K),
        ]
        # TODO2：RRF融合两路排名
        fused = rrf_fuse(rankings, RRF_K)
        # TODO3：融合结果截取候选池
        candidates = fused[:CANDIDATE_K]
        # TODO4：rerank精排，最终返回top_k条（入参top_k）
        return reranker.rerank(question, candidates, top_k)
    return retriever

