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
