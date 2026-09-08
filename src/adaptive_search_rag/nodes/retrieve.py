from adaptive_search_rag.state import RAGState


def build_retriever():
    # ===== 硬编码配置，务必和 ingest.py 里那三行完全一致 =====
    #向量数据库磁盘存放路径
    CHROMA_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    COLLECTION_NAME = "adaptive_knowledge"
    # ========================================================
    #提供很多已经封装好的向量化工具
    from chromadb.utils import embedding_functions
    import chromadb
    #传字符串列表进去，输出向量列表
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )
    #把向量数据库全部数据保存到磁盘文件夹
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    #**获取一个已经存在的集合**。
    collection = client.get_collection(name=COLLECTION_NAME)

    def retriever(question: str, top_k: int = 3) -> list[str]:
        #传入用户问题,返回3个相似的块
        #把用户问题向量化
        query_emb = ef([question])

        # TODO 2：调用 ChromaDB 的 query 接口，传入向量化问题，返回最相似的 top_k 个块
        #**query做向量相似度查询的核心方法**。
        res = collection.query(
            query_embeddings=query_emb,
            n_results=top_k
        )

        # TODO 3：从结果里取出文档文本，注意嵌套结构，返回 list[str]
        # res["documents"] 是 [[doc1,doc2,doc3]] 两层嵌套
        #拿第一个查询对应的内层列表**
        docs = res["documents"][0]
        return docs

    return retriever
def retrieve_node(state: RAGState, top_k: int = 3, retriever=None) -> dict:
    """检索节点：从向量库取 top_k 片段，返回统一 {"contexts": [...]}"""
    if retriever is None:                 # 真实运行：造真的
        retriever = build_retriever()
    question = state["question"]
    contexts = retriever(question, top_k)
    return {"contexts": contexts}

