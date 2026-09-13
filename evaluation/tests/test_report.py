from evaluation.report import diff_runs, summarize


def _c(i, t, scores=None, branch_actual=None,
       branch_expected=None, grade="correct"):
    return {"id": i, "type": t, "scores": scores or {}, "grade": grade,
            "retry_count": 0, "branch_actual": branch_actual,
            "branch_expected": branch_expected}


def test_summarize_excludes_oob_from_quality_metrics():
    cases = [_c("kb-01", "kb", {"faithfulness": 1.0}),
             _c("oob-01", "oob", {"faithfulness": 0.0})]
    s = summarize(cases)
    # oob 的 0.0 绝不能被平均进来 —— 这是本设计的核心不变量
    assert s["overall"]["faithfulness"] == 1.0


def test_summarize_groups_by_type():
    cases = [_c("kb-01", "kb", {"faithfulness": 0.8}),
             _c("kb-02", "kb", {"faithfulness": 0.6}),
             _c("oob-01", "oob")]
    s = summarize(cases)
    assert s["by_type"]["kb"]["n"] == 2
    assert s["by_type"]["kb"]["faithfulness"] == 0.7
    assert s["by_type"]["oob"]["n"] == 1
    assert "faithfulness" not in s["by_type"]["oob"]   # oob 不出质量分


def test_summarize_branch_match_rate():
    cases = [_c("r-01", "rewrite",
                branch_actual="generate",
                branch_expected="generate"),
             _c("r-02", "rewrite",
                branch_actual="search",
                branch_expected="generate")]
    s = summarize(cases)
    assert s["by_type"]["rewrite"]["branch_match"] == 0.5


def test_summarize_ignores_cases_without_expectation():
    cases = [_c("kb-01", "kb",
                branch_actual="generate", branch_expected=None)]
    s = summarize(cases)
    assert "branch_match" not in s["by_type"]["kb"]


def test_diff_runs_reports_change():
    old = {"run_id": "old", "summary": {"overall": {"faithfulness": 0.5}}}
    new = {"run_id": "new", "summary": {"overall": {"faithfulness": 0.8}}}
    out = diff_runs(old, new)
    assert "faithfulness" in out
    assert "0.50" in out and "0.80" in out


def test_summarize_skips_none_scores():
    """打分失败的题记为 None：不能当 0 算进均值，也不能让 sum() 炸。"""
    cases = [_c("kb-01", "kb", {"faithfulness": 0.8}),
             _c("kb-02", "kb", {"faithfulness": None})]
    s = summarize(cases)
    assert s["by_type"]["kb"]["faithfulness"] == 0.8     # 只算成功的那道
    assert s["overall"]["faithfulness"] == 0.8


# ---- 走 search 分支的题必须排除出质量分（spec §6 的规则修正）----


def test_summarize_excludes_search_branch_from_quality_metrics():
    """走 search 的题 contexts 被桩替换过，质量分评的是桩不是系统 —— 必须排除。"""
    cases = [_c("kb-01", "kb", {"faithfulness": 1.0}, branch_actual="generate"),
             _c("kb-02", "kb", {"faithfulness": 0.0}, branch_actual="search")]
    s = summarize(cases)
    assert s["by_type"]["kb"]["n"] == 2          # 题数还是 2
    assert s["by_type"]["kb"]["scored"] == 1     # 但只有 1 道参与质量分
    assert s["by_type"]["kb"]["faithfulness"] == 1.0


def test_summarize_overall_also_excludes_search_branch():
    cases = [_c("kb-01", "kb", {"faithfulness": 1.0}, branch_actual="generate"),
             _c("kb-02", "kb", {"faithfulness": 0.0}, branch_actual="search")]
    s = summarize(cases)
    assert s["overall"]["faithfulness"] == 1.0


def test_quality_eligible_ignores_cases_without_branch_info():
    """branch_actual 缺失（老档案）时不能误排除。"""
    cases = [_c("kb-01", "kb", {"faithfulness": 0.8})]   # branch_actual=None
    s = summarize(cases)
    assert s["by_type"]["kb"]["scored"] == 1
    assert s["by_type"]["kb"]["faithfulness"] == 0.8


def test_scored_is_zero_for_oob():
    """oob 一道都不参与质量分 —— 这个 0 必须显示出来，不能静默丢弃。"""
    cases = [_c("oob-01", "oob", None, branch_actual="search")]
    s = summarize(cases)
    assert s["by_type"]["oob"]["n"] == 1
    assert s["by_type"]["oob"]["scored"] == 0
    assert "faithfulness" not in s["by_type"]["oob"]
