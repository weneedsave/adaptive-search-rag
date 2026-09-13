"""汇总表与基线对比。纯函数，不依赖 LLM。"""
from __future__ import annotations
from collections import defaultdict
from evaluation.golden import QUALITY_TYPES


def summarize(run_cases: list[dict]) -> dict:
    """TODO 1：返回 {"by_type": {type: {...}}, "overall": {...}}。
    规则：
      - 每个 type 出：n（该题型题数）、各质量指标均值、
          branch_match（**仅当**该题型有题带 branch_expected 时才出这个键）
      - 质量指标均值**只统计 type ∈ QUALITY_TYPES 的题**（oob 排除）
      - overall 也只统计 QUALITY_TYPES 的题
      - scores 为空的题不参与均值，**不要把空当成 0 算进去**
    已知局限：小样本下均值抖动大，13 题不是 benchmark。
    """
    groups = defaultdict(list)
    for case in run_cases:
        groups[case["type"]].append(case)

    by_type = {}
    # 按题型分组计算指标
    for typ, case_list in groups.items():
        item = {"n": len(case_list)}
        # 计算路由匹配率 branch_match
        has_expected = any(c.get("branch_expected") is not None for c in case_list)
        if has_expected:
            match_count = sum(1 for c in case_list if c["branch_actual"] == c["branch_expected"])
            item["branch_match"] = match_count / len(case_list)

        # 仅QUALITY_TYPES计算RAGAS质量分
        if typ in QUALITY_TYPES:
            score_list: list[dict] = [c["scores"] for c in case_list if c.get("scores")]
            if score_list:
                metric_names = score_list[0].keys()
                for metric in metric_names:
                    vals = [s[metric] for s in score_list if metric in s]
                    item[metric] = sum(vals) / len(vals)
        by_type[typ] = item

    # overall全局汇总，只统计QUALITY_TYPES
    overall = defaultdict(list)
    for case in run_cases:
        if case["type"] not in QUALITY_TYPES:
            continue
        s = case.get("scores", {})
        if not s:
            continue
        for k, v in s.items():
            overall[k].append(v)
    overall_avg = {k: sum(v) / len(v) for k, v in overall.items()}

    return {
        "by_type": by_type,
        "overall": overall_avg
    }


def format_summary(summary: dict) -> str:
    """按 spec §9.2 的表格返回字符串。

    oob 走桩，其质量分不参与统计（summarize 已跳过），
    四个质量分列显示 '—'。
    """
    W_TYPE, W_N, W_BM, W_M, W_CP = 10, 6, 14, 14, 16
    line = "-" * (W_TYPE + W_N + W_BM + W_M * 2 + W_CP + W_M)

    def cell(value) -> str:
        """None（该题型没有这项数据）显示 '—'，数值显示两位小数。

        注意 0.0 要显示成 '0.00' 而不是 '—' ——
        「没有期望」和「符合率 0%」是两回事。
        """
        return f"{value:.2f}" if isinstance(value, (int, float)) else "—"

    lines = [
        "type".ljust(W_TYPE) + "n".ljust(W_N) + "branch_match".ljust(W_BM)
        + "faithfulness".ljust(W_M) + "answer_rel".ljust(W_M)
        + "ctx_precision".ljust(W_CP) + "ctx_recall".ljust(W_M),
        line,
    ]
    for typ, data in summary["by_type"].items():
        lines.append(
            typ.ljust(W_TYPE) + str(data["n"]).ljust(W_N)
            + cell(data.get("branch_match")).ljust(W_BM)
            + cell(data.get("faithfulness")).ljust(W_M)
            + cell(data.get("answer_relevancy")).ljust(W_M)
            + cell(data.get("context_precision")).ljust(W_CP)
            + cell(data.get("context_recall")).ljust(W_M)
        )

    lines.append(line)
    ov = summary["overall"]
    lines.append(
        "OVERALL".ljust(W_TYPE + W_N)
        + cell(ov.get("faithfulness")).ljust(W_M)
        + cell(ov.get("answer_relevancy")).ljust(W_M)
        + cell(ov.get("context_precision")).ljust(W_CP)
        + cell(ov.get("context_recall")).ljust(W_M)
    )
    return "\n".join(lines)


def diff_runs(old: dict, new: dict) -> str:
    """TODO 3：对比两份 run 的 summary["overall"]，返回带 ↑↓ 的 diff 表字符串。
    分数保留**两位小数**（测试断言里有 "0.50" / "0.80"）。
    """
    old_over = _get_summary(old)["overall"]
    new_over = _get_summary(new)["overall"]
    lines = []
    lines.append(f"{'metric':<16}{'old':<8}{'new':<8}{'delta'}")
    lines.append("-" * 40)
    all_metrics = set(old_over.keys()) | set(new_over.keys())
    for metric in all_metrics:
        o_val = old_over.get(metric)
        n_val = new_over.get(metric)
        if o_val is None or n_val is None:
            continue
        delta = n_val - o_val
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        lines.append(f"{metric:<16}{o_val:.2f}{'':<2}{n_val:.2f}{'':<2}{delta:.2f} {arrow}")
    return "\n".join(lines)


def _get_summary(run: dict) -> dict:

    return run.get("summary") or summarize(run["cases"])