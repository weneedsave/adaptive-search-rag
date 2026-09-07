from pathlib import Path
from adaptive_search_rag.ingest import load_markdown, split_text, ingest_directory

def test_split_text_has_overlap():
    text = "字" * 1200          # 1200字，chunk_size=500 → 至少切出 3 块
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    # 关键断言：第一块的最后50字 == 第二块的前50字（这就是"重叠"）
    assert chunks[0][-50:] == chunks[1][:50]

def test_load_markdown(tmp_path):
    content = "# 标题\n这是测试内容"
    md_file = tmp_path / "test.md"
    md_file.write_text(content, encoding="utf-8")   # 写一个临时 .md
    result = load_markdown(str(md_file))
    assert result == content          # 或 assert "测试内容" in result
    assert "测试内容" in result

def test_ingest_directory(tmp_path):
    md_file = tmp_path / "doc.md"
    md_file.write_text("RAG 原理。" * 300, encoding="utf-8")   # 足够长，能切出多块
    # 用临时 collection_name，避免污染主库 adaptive_knowledge
    count = ingest_directory(str(tmp_path), collection_name="test_collection")
    assert count > 0

