# evaluation/tests/test_golden.py
import json
from evaluation.golden import (
    BLOCK_TYPES,
    MUST_REFUSE,
    QUALITY_TYPES,
    GoldenCase,
    load_golden,
    validate_golden,
)


def _write(tmp_path, rows):
    p = tmp_path / "g.jsonl"
    p.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8"
    )
    return p


def test_load_parses_fields(tmp_path):
    p = _write(tmp_path, [
        {
            "id": "kb-01",
            "type": "kb",
            "question": "什么是RAG？",
            "reference": "检索增强生成",
            "expect_branch": "generate",
            "note": "基础"
        }
    ])
    cases = load_golden(p)
    assert len(cases) == 1
    c = cases[0]
    assert (c.id, c.type, c.expect_branch, c.note) == ("kb-01", "kb", "generate", "基础")


def test_expect_branch_defaults_to_none(tmp_path):
    p = _write(tmp_path, [
        {"id": "kb-01", "type": "kb", "question": "q", "reference": "r"}
    ])
    assert load_golden(p)[0].expect_branch is None


def test_oob_must_not_be_in_quality_types():
    # 核心不变量锁死
    assert "oob" not in QUALITY_TYPES
    assert set(QUALITY_TYPES) < set(BLOCK_TYPES)


def test_validate_accepts_good_set():
    cases = [
        GoldenCase("kb-01", "kb", "q", "r", "generate"),
        GoldenCase("oob-01", "oob", "q", MUST_REFUSE, "search")
    ]
    assert validate_golden(cases) == []


def test_validate_rejects_duplicate_id():
    cases = [
        GoldenCase("kb-01", "kb", "q", "r"),
        GoldenCase("kb-01", "kb", "q2", "r2")
    ]
    errs = validate_golden(cases)
    assert any("重复" in e for e in errs)


def test_validate_rejects_bad_type_and_branch():
    cases = [
        GoldenCase("x-01", "nonsense", "q", "r", "fly")
    ]
    errs = validate_golden(cases)
    assert any("type" in e for e in errs)
    assert any("expect_branch" in e for e in errs)


def test_validate_rejects_empty_question_and_reference():
    cases = [
        GoldenCase("x-01", "kb", "", "")
    ]
    errs = validate_golden(cases)
    assert any("question" in e for e in errs)
    assert any("reference" in e for e in errs)


def test_validate_enforces_oob_reference():
    cases = [
        GoldenCase("oob-01", "oob", "q", "随便写的答案")
    ]
    errs = validate_golden(cases)
    assert any("oob" in e for e in errs)
