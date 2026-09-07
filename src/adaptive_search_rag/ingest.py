# src/adaptive_search_rag/ingest.py
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions


def load_markdown(path: str) -> str:
    """读一个 .md 文件，返回纯文本"""
    # TODO 1：UTF-8 编码打开文件，读全文返回
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """把长文本切成若干块，相邻块重叠 chunk_overlap 字"""
    chunks = []
    start = 0
    text_len = len(text)
    # TODO 2：步长 = 块大小 - 重叠大小
    step = chunk_size - chunk_overlap

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += step

    return chunks


def ingest_directory(dir_path: str, collection_name: str = "adaptive_knowledge") -> int:
    """遍历目录下所有 .md 文件，分块、向量化、入库，返回入库 块 总数"""
    # ========== 硬编码配置（阶段1先用，后续换 config 替换这里即可） ==========
    CHROMA_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    # ==========================================================================

    # 向量化工具
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )

    # 持久化客户端：数据写入磁盘文件夹
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # 获取或新建集合，绑定 embedding 函数, 用于向量化
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef #如果新建集合就为新建的集合绑定这个向量化函数
    )

    # 扫描目标目录下所有 .md 文件
    dir_p = Path(dir_path)
    #匹配查找文件
    md_files = list(dir_p.glob("*.md"))

    total_chunks = 0

    for md_file in md_files:
        # 1. 读取文件全文
        file_text = load_markdown(str(md_file))
        # 2. 切分成带重叠的文本块
        chunks = split_text(file_text)

        documents = []
        ids = []
        for idx, chunk in enumerate(chunks):
            # 生成唯一ID：文件名#序号
            doc_id = f"{md_file.name}#{idx}"
            ids.append(doc_id)
            documents.append(chunk)

        if documents:
            # 3. 批量入库（Chroma 内部自动向量化）
            collection.upsert(
                documents=documents,
                ids=ids
            )
            total_chunks += len(documents)

    return total_chunks
