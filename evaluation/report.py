"""汇总表与基线对比。纯函数，不依赖 LLM。"""
from __future__ import annotations
from collections import defaultdict
from evaluation.golden import QUALITY_TYPES


def is_quality_eligible(case: dict) -> bool:
    """判断这道题的质量分是否可解释。

    两个条件缺一不可（spec §6）：
      1. type ∈ QUALITY_TYPES —— oob 注定走 search，本来就不该算
      2. branch_actual != "search" —— 走 search 的题，contexts 被桩替换过，
         质量分评的是桩，不是你的系统

    branch_actual 缺失时（老档案、或没跑过图的构造数据）视为合格，
    不做排除，避免误伤 —— 这是 .get 而不是下标访问的原因。
    """
    if case.get("type") not in QUALITY_TYPES:
        return False
    return case.get("branch_actual") != "search"


def summarize(run_cases: list[dict]) -> dict:
    """返回 {"by_type": {type: {...}}, "overall": {...}}。

    规则：
      - 每个 type 出：n（该题型题数）、scored（其中实际参与质量分的题数）、
          各质量指标均值、branch_match（**仅当**该题型有题带 branch_expected 时才有）
      - 质量分**只统计 is_quality_eligible 为真的题**：
        type ∈ QUALITY_TYPES **且** branch_actual != "search"
      - overall 同样只统计 is_quality_eligible 为真的题
      - scored 为空的题（或某指标为 None）不参与该指标均值，
        **不要把空当成 0 算进去**

    已知局限：小样本下均值抖动大（13 题不是 benchmark），
    且裁判 LLM 有 ±0.03 量级的固有噪声（spec §11），
    小于 ~0.05 的差异不可当信号解读。
    """
    groups = defaultdict(list)
    for case in run_cases:
        groups[case["type"]].append(case)

    by_type = {}
    # 按题型分组计算指标
    for typ, case_list in groups.items():
        item = {"n": len(case_list)}
        # 实际参与质量分的题数。放在 if 外面：连 oob 组也要显示 scored=0，
        # 否则「被排除」就成了隐形的，静默丢弃（spec §6 明令禁止）。
        eligible = [c for c in case_list if is_quality_eligible(c)]
        item["scored"] = len(eligible)

        # 计算路由匹配率 branch_match
        has_expected = any(c.get("branch_expected") is not None for c in case_list)
        if has_expected:
            match_count = sum(1 for c in case_list if c["branch_actual"] == c["branch_expected"])
            item["branch_match"] = match_count / len(case_list)

        # RAGAS质量分：只算 is_quality_eligible 的题（类型合格 且 没走 search）
        score_list: list[dict] = [c["scores"] for c in eligible if c.get("scores")]
        if score_list:
            metric_names = score_list[0].keys()
            for metric in metric_names:
                # None = 该题该指标打分失败（见 metrics._score_all），跳过不参与均值
                vals = [s[metric] for s in score_list if s.get(metric) is not None]
                if vals:
                    item[metric] = sum(vals) / len(vals)
        by_type[typ] = item

    # overall 全局汇总：与 by_type 用同一套合格判据（is_quality_eligible）。
    # 保持纯指标均值 —— scored 计数由 format_summary 从 by_type 汇总，
    # 不混进来，否则 diff_runs 会把题数当分数格式化。
    overall = defaultdict(list)
    for case in run_cases:
        if not is_quality_eligible(case):
            continue
        for k, v in case.get("scores", {}).items():
            if v is not None:      # 同 by_type：失败记为 None，不参与均值
                overall[k].append(v)
    overall_avg = {k: sum(v) / len(v) for k, v in overall.items()}

    return {
        "by_type": by_type,
        "overall": overall_avg
    }


def format_summary(summary: dict) -> str:
    """按 spec §9.2 的表格返回字符串。

    scored 列 = 该题型里实际参与质量分的题数（n 减去被排除的）。
    被排除的题数必须看得见，不能静默丢弃（spec §6）。
    质量分列显示 '—' 表示「该组没有可评分的题」，不是「0 分」。
    """
    W_TYPE, W_N, W_S, W_BM, W_M, W_CP = 10, 6, 8, 14, 14, 16
    line = "-" * (W_TYPE + W_N + W_S + W_BM + W_M * 2 + W_CP + W_M)

    def cell(value) -> str:
        """None（该题型没有这项数据）显示 '—'，数值显示两位小数。

        注意 0.0 要显示成 '0.00' 而不是 '—' ——
        「没有期望」和「符合率 0%」是两回事。
        """
        return f"{value:.2f}" if isinstance(value, (int, float)) else "—"

    lines = [
        "type".ljust(W_TYPE) + "n".ljust(W_N) + "scored".ljust(W_S)
        + "branch_match".ljust(W_BM) + "faithfulness".ljust(W_M)
        + "answer_rel".ljust(W_M) + "ctx_precision".ljust(W_CP)
        + "ctx_recall".ljust(W_M),
        line,
    ]
    scored_total = 0
    for typ, data in summary["by_type"].items():
        scored_total += data.get("scored", 0)
        lines.append(
            typ.ljust(W_TYPE) + str(data["n"]).ljust(W_N)
            + str(data.get("scored", 0)).ljust(W_S)
            + cell(data.get("branch_match")).ljust(W_BM)
            + cell(data.get("faithfulness")).ljust(W_M)
            + cell(data.get("answer_relevancy")).ljust(W_M)
            + cell(data.get("context_precision")).ljust(W_CP)
            + cell(data.get("context_recall")).ljust(W_M)
        )

    lines.append(line)
    ov = summary["overall"]
    lines.append(
        "OVERALL".ljust(W_TYPE + W_N) + str(scored_total).ljust(W_S)
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