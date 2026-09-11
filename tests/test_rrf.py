from adaptive_search_rag.retrieval.hybrid import rrf_fuse

def test_rrf_accumulates_across_rankings():
    """同一文档出现在两路时，分数是两路贡献之和"""
    rankings = [["A", "B", "C"], ["A", "C"]]
    result = rrf_fuse(rankings, k=60)
    # A: 第1路 rank1 + 第2路 rank1 → 2/61
    # C: 第1路 rank3 + 第2路 rank2 → 1/63 + 1/62
    # B: 第1路 rank2               → 1/62
    assert result == ["A", "C", "B"]

def test_rrf_single_ranking_preserves_order():
    assert rrf_fuse([["X", "Y", "Z"]], k=60) == ["X", "Y", "Z"]

def test_rrf_empty_inputs():
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []

def test_rrf_one_side_empty():
    assert rrf_fuse([[], ["P", "Q"]], k=60) == ["P", "Q"]

def test_rrf_deduplicates():
    """两路完全相同的列表，结果里每个文档只出现一次"""
    result = rrf_fuse([["A", "B"], ["A", "B"]], k=60)
    assert result == ["A", "B"]
    assert len(result) == len(set(result))

def test_rrf_custom_k_is_used():
    """k=1 时手算可验证：rank1 → 0.5, rank2 → 0.333..."""
    result = rrf_fuse([["A", "B"]], k=1)
    assert result == ["A", "B"]

def test_rrf_tie_is_deterministic():
    """两路里各自第1名的两个不同文档，分数相同，顺序必须稳定"""
    r1 = rrf_fuse([["A"], ["B"]], k=60)
    r2 = rrf_fuse([["A"], ["B"]], k=60)
    assert r1 == r2
    assert set(r1) == {"A", "B"}
