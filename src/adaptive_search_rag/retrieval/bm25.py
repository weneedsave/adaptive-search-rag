"""中文 BM25 检索：jieba 分词 + rank_bm25

BM25 是词频统计型稀疏检索，和向量检索互补：
- 向量检索擅长语义相近，但对专有名词/缩写/编号不敏感
- BM25 擅长精确词命中，但不懂同义改写
两者融合即为工业标准的「混合检索」。
"""
import re

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """中文分词。jieba.cut 返回生成器，list()转为列表
    过滤空白、纯标点符号token，避免污染BM25的IDF计算
    """
    # 匹配：全部由标点/符号构成的token（中文+英文标点）
    punc_pattern = re.compile(r"^[^\u4e00-\u9fa5a-zA-Z0-9]+$")

    tokens = list(jieba.cut(text))
    valid_tokens = []
    for t in tokens:
        # 1. 跳过空白（空格、换行、tab）
        if not t.strip():
            continue
        # 2. 跳过纯标点符号
        if punc_pattern.fullmatch(t):
            continue
        valid_tokens.append(t)
    return valid_tokens


#构建一个知识库检索器
def build_bm25_retriever(chunks: list[str]):
    if not chunks:
        def empty_retriever(question: str, top_k: int = 10) -> list[str]:
            return []
        return empty_retriever

    corpus = [tokenize(chunk) for chunk in chunks]#切一个块,切成词
    #构建关键词索引 + 计算关键词匹配分数
    bm25 = BM25Okapi(corpus)

    def retriever(question: str, top_k: int = 10) -> list[str]:
        query_tokens = tokenize(question)
        scores = bm25.get_scores(query_tokens)#遍历全部知识库,返回一个得分
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)#拿到得分,降序排序
        top_indices = [idx for idx, score in indexed_scores[:top_k]]
        return [chunks[idx] for idx in top_indices]
    return retriever

