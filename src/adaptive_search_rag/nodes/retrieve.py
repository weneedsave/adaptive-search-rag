from functools import lru_cache
import chromadb
from chromadb.utils import embedding_functions

from adaptive_search_rag.retrieval.bm25 import build_bm25_retriever
from adaptive_search_rag.retrieval.hybrid import build_hybrid_retriever
from adaptive_search_rag.retrieval.rerank import CrossEncoderReranker, NoOpReranker
from adaptive_search_rag.state import RAGState

#向量检索
def build_vector_retriever():
    """建 Chroma collection，返回纯向量检索闭包"""
    # ===== 硬编码配置，务必和 ingest.py 里那三行完全一致 =====
    CHROMA_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    COLLECTION_NAME = "adaptive_knowledge"
    # ========================================================
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        local_files_only=True,          # 同 ingest，HF被墙必须离线
    )
    # 持久化客户端：数据写入磁盘文件夹
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    #取出COLLECTION_NAME里的全部块
    collection = client.get_collection(name=COLLECTION_NAME)

    def retriever(question: str, top_k: int = 3) -> list[str]:
        # TODO 1：用 ef 把 question 向量化（ef吃 list[str] 吐 list[list[float]]）
        query_emb = ef([question])
        #collection.query输入向量，找相似文档，返回 top_k 片段
        res = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )
        # TODO 3：从结果里取出文档文本，注意嵌套结构，返回 list[str]
        # res["documents"] 是 [[doc1,doc2,doc3]] 两层嵌套
        docs = res["documents"][0]
        return docs
    return retriever


@lru_cache(maxsize=1)
def build_retriever():
    """组装混合检索：向量 + BM25 双路粗筛 → RRF 融合 → CrossEncoder 精排
    @lru_cache(maxsize=1)：进程内只构建一次。否则CLI每问一次都要重载
    Embedding + CrossEncoder 模型（实测CrossEncoder加载要4秒）。
    长驻进程若更新了语料，需调用 build_retriever.cache_clear()。
    """
    # 1. 构造向量检索器
    vector_retriever = build_vector_retriever()

    # 2. 全量拉取Chroma里的文档chunk，构建BM25索引
    #PersistentClient链接到chroma_db
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="adaptive_knowledge")
    # collection.get()取documents这一个列表(知识库文本块)
    chunks = collection.get()["documents"]

    # 注释：collection.get() 全量拉取所有文档，返回一维list[str]


    #传入知识库块返回bm25_retriever函数
    bm25_retriever = build_bm25_retriever(chunks)

    # 4. 构建重排器（带降级）
    reranker = _build_reranker()

    # 5. 组装混合检索流水线并返回
    return build_hybrid_retriever(vector_retriever, bm25_retriever, reranker)

#重排,两种检索完成后使用模型进行仔细重排
def _build_reranker():
    """加载 CrossEncoder；失败则降级 NoOpReranker（保住 RRF 顺序，链路不断）"""
    # CrossEncoderReranker为懒加载：实例化对象本身不加载大模型
    reranker = CrossEncoderReranker()
    try:
        reranker._get_model()     # 显式触发模型加载，真正载入权重
        return reranker
    except Exception as e:
        print(f"[警告] CrossEncoder 加载失败，已降级为 NoOpReranker：{e}")
        return NoOpReranker()


def retrieve_node(state: RAGState, top_k: int = 3, retriever=None) -> dict:
    """检索节点：从混合检索取 top_k 片段，返回统一 {"contexts": [...]}"""
    if retriever is None:                 # 真实运行：造真的
        retriever = build_retriever()
    question = state["question"]
    contexts = retriever(question, top_k)
    return {"contexts": contexts}
