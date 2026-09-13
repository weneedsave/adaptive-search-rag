from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from evaluation.golden import load_golden
from evaluation.metrics import (
    build_embeddings,
    build_judge_llm,
    build_metrics,
    score,
)
from evaluation.stubs import build_eval_graph, contains_stub

GOLDEN = Path("evaluation/datasets/golden.jsonl")
RUNS_DIR = Path("evaluation/runs")

# 评测不用 checkpointer，但保持同一份定义，避免和生产路径分叉。
INITIAL_STATE = {"question": "", "retrieval_query": "", "contexts": [],
                 "grade": "", "retry_count": 0, "answer": ""}


def detect_branch(state: dict, contexts: list[str]) -> str:
    """TODO 1：判定实际走了哪条分支（按"最远"的报）。
    contexts 含桩标记    → "search"
    否则 retry_count > 0 → "rewrite"
    否则                 → "generate"
    提示：桩标记检测用 contains_stub(contexts)。
    docstring里写明已知局限：incorrect→rewrite→…→search 只报"search"，
    会掩盖中间那次 retry。
    """
    if contains_stub(contexts):
        return "search"
    if state.get("retry_count", 0) > 0:
        return "rewrite"
    return "generate"


def run_case(case, app) -> dict:
    """TODO 2：单题跑图。
    state = {**INITIAL_STATE, "question": case.question}
    result = app.invoke(state)
    返回 dict：
      {"id": case.id, "type": case.type, "question": case.question,
       "answer": result["answer"],
       "contexts": result.get("contexts", []),
       "grade": result.get("grade", ""),
       "retry_count": result.get("retry_count", 0),
       "branch_actual": detect_branch(result, result.get("contexts", [])),
       "branch_expected": case.expect_branch,
       "scores": {}}
    """
    # 合并初始状态，注入当前问题
    state = {**INITIAL_STATE, "question": case.question}
    result = app.invoke(state)
    branch_actual = detect_branch(result, result.get("contexts", []))
    return {
        "id": case.id,
        "type": case.type,
        "question": case.question,
        "answer": result["answer"],
        "contexts": result.get("contexts", []),
        "grade": result.get("grade", ""),
        "retry_count":  result.get("retry_count", 0),
        "branch_actual": branch_actual,
        "branch_expected": case.expect_branch,
        "scores": {}
    }


def run_all(cases, app, llm=None, emb=None) -> dict:
    """TODO 3：跑全部题目，组装 run dict。
    1. run_cases = [run_case(c, app) for c in cases]
    2. 若 llm 不为 None：
         metrics = build_metrics(llm, emb)
         scored  = score(cases, run_cases, metrics)
   # {case_id: {指标: 分}}
         把 scored 按 id 填回每题的 "scores"（没分保持 {}）
    3. git_commit：subprocess 跑 git rev-parse --short HEAD，失败填 "unknown"
    4. run_id：datetime.now().strftime("%Y-%m-%d-%H%M")
    5. 返回 {"run_id", "git_commit", "config", "cases": run_cases}
       config 先硬编码 {"reranker": "cross-encoder", "top_k": 3}
    """
    # 1. 遍历所有样例，逐个跑CRAG图
    run_cases = [run_case(c, app) for c in cases]

    # 2. llm不为None才执行RAGAS打分
    if llm is not None:
        metrics = build_metrics(llm, emb)
        scored = score(cases, run_cases, metrics)
        # 将分数回填到每个run_case的scores字段
        for rc in run_cases:
            rc["scores"] = scored.get(rc["id"], {})

    # 3. 获取git短commit哈希
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    # 4. 生成run_id，用当前时间
    run_id = datetime.now().strftime("%Y-%m-%d-%H%M")

    # 5. 打包返回结果
    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "config": {"reranker": "cross-encoder", "top_k": 3},
        "cases": run_cases
    }


def save_run(run: dict, runs_dir: Path = RUNS_DIR) -> Path:
    """TODO 4：写到 runs_dir/<run_id>.json。
    mkdir(parents=True, exist_ok=True)
    ensure_ascii=False, indent=2
    返回写入的 Path。
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{run['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    return out_path


def print_behavior_table(run: dict) -> None:
    """TODO 5：简单打印每题的行为指标。
    每题一行：id / type / grade / retry_count /
             branch_actual / branch_expected / scores
    （Task 8 会把这个换成 report.format_summary 出的漂亮表格。）
    """
    print(f"{'id':<10}{'type':<10}{'grade':<8}{'retry':<8}{'actual':<12}{'expected':<12} scores")
    print("-" * 80)
    for item in run["cases"]:
        sid = item["id"]
        typ = item["type"]
        grade = item["grade"]
        retry = item["retry_count"]
        act = item["branch_actual"]
        exp = item["branch_expected"]
        scores = item["scores"]
        print(f"{sid:<10}{typ:<10}{grade:<8}{retry:<8}{act:<12}{exp:<12} {scores}")


def main(argv=None):
    """TODO 6：argparse。
      --no-llm   只跑行为指标，跳过 RAGAS（省钱、快）
      默认       跑一轮含打分
    流程：load_golden(GOLDEN) → build_eval_graph() → run_all(...)
          → save_run(...) → print_behavior_table(...)
    --no-llm 时 llm 和 emb 都传 None（别去 build，会白加载模型）。
    """
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="CRAG评测主程序")
    parser.add_argument("--no-llm", action="store_true", help="跳过RAGAS打分，仅跑图收集行为数据")
    args = parser.parse_args(argv)

    # 加载golden评测数据集
    cases = load_golden(GOLDEN)
    # 构建评测用LangGraph图（带桩节点）
    app = build_eval_graph()

    if args.no_llm:
        llm = None
        emb = None
    else:
        llm = build_judge_llm()
        emb = build_embeddings()

    # 执行全部样例
    run_result = run_all(cases, app, llm=llm, emb=emb)
    # 保存json结果文件
    saved_path = save_run(run_result)
    print(f"\n✅ 评测结果保存至：{saved_path}")
    # 打印表格
    print_behavior_table(run_result)


if __name__ == "__main__":
    main()
