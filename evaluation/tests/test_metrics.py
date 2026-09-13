import pytest
from evaluation.golden import GoldenCase
from evaluation.metrics import METRIC_NAMES, metric_args, select_pairs


def _case(i, t):
    return GoldenCase(i, t, f"q{i}", f"r{i}")


def _rc(i, t):
    return {"id": i, "type": t, "question": f"q{i}", "answer": f"a{i}",
            "contexts": [f"c{i}"]}


def test_metric_names_are_the_four_we_use():
    assert METRIC_NAMES == ("faithfulness",
                            "answer_relevancy",
                            "context_recall",
                            "context_precision")


def test_select_pairs_excludes_oob():
    cases = [_case("kb-01", "kb"), _case("oob-01", "oob")]
    rcs = [_rc("kb-01", "kb"), _rc("oob-01", "oob")]
    pairs = select_pairs(cases, rcs)
    # oob 走桩，质量分评的是桩不是系统 —— 必须排除
    assert len(pairs) == 1
    assert pairs[0][0].id == "kb-01"


def test_select_pairs_keeps_all_quality_types():
    cases = [_case("kb-01", "kb"), _case("h-01", "kb-hard"), _case("r-01", "rewrite")]
    rcs = [_rc(c.id, c.type) for c in cases]
    assert [c.id for c, _ in select_pairs(cases, rcs)] == ["kb-01", "h-01", "r-01"]


def test_select_pairs_skips_missing_run_case():
    assert select_pairs([_case("kb-01", "kb")], []) == []


def test_faithfulness_args():
    a = metric_args("faithfulness", _case("kb-01", "kb"), _rc("kb-01", "kb"))
    assert set(a) == {"user_input", "response", "retrieved_contexts"}
    assert a["user_input"] == "qkb-01"
    assert a["response"] == "akb-01"
    assert a["retrieved_contexts"] == ["ckb-01"]


def test_answer_relevancy_takes_no_contexts():
    a = metric_args("answer_relevancy", _case("kb-01", "kb"), _rc("kb-01", "kb"))
    assert set(a) == {"user_input", "response"}
    # 实测签名只有这两个


def test_context_recall_takes_reference_not_response():
    a = metric_args("context_recall", _case("kb-01", "kb"), _rc("kb-01", "kb"))
    assert set(a) == {"user_input", "retrieved_contexts", "reference"}
    assert a["reference"] == "rkb-01"


def test_context_precision_args():
    a = metric_args("context_precision", _case("kb-01", "kb"), _rc("kb-01", "kb"))
    assert set(a) == {"user_input", "response", "retrieved_contexts"}


def test_metric_args_rejects_unknown_name():
    with pytest.raises(KeyError):
        metric_args("nonsense", _case("kb-01", "kb"), _rc("kb-01", "kb"))
