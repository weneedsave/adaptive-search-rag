from __future__ import annotations
import argparse
import json
import os
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
# ========= (a) 新增导入report模块 =========
from evaluation.report import diff_runs, format_summary, summarize

GOLDEN = Path("evaluation/datasets/golden.jsonl")
RUNS_DIR = Path("evaluation/runs")

# 评测不用 checkpointer，但保持同一份定义，避免和生产路径分叉。
INITIAL_STATE = {"question": "", "retrieval_query": "", "contexts": [],
                 "grade": "", "retry_count": 0, "answer": ""}


def detect_branch(state: dict, contexts: list[str]) -> str:
    """TODO 1：判定实际走了哪条分支（按"最远"的报,判断crag做的决定是不是真的被采纳,
    contexts 含桩标记    → "search"
    否则 retry_count > 0 → "rewrite"
    否则                 → "generate"
    提示：桩标记检测用 contains_stub(contexts)。
    docstring里写明已知局限：incorrect→rewrite→…→search 只报"search"，
    会掩盖中间那次 retry。
    """
    #遍历检索返回的文档列表，看里面有没有**search 桩标记文本**
    if contains_stub(contexts):
        return "search"
    if state.get("retry_count", 0) > 0:
        return "rewrite"
    return "generate"


MAX_RUN_ATTEMPTS = 3


def _failed_case(case, err) -> dict:
    """跑图彻底失败时的占位记录。

    grade="error" 会被 report.is_quality_eligible 排除出质量分 ——
    没跑起来的题，没有任何可解释的分数。
    """
    return {
        "id": case.id,
        "type": case.type,
        "question": case.question,
        "answer": "",
        "contexts": [],
        "grade": "error",
        "retry_count": 0,
        "branch_actual": None,
        "branch_expected": case.expect_branch,
        "scores": {},
        "error": f"{type(err).__name__}: {err}",
    }


def run_case(case, app, max_attempts: int = MAX_RUN_ATTEMPTS) -> dict:
    """处理一道评测题目，跑一遍 CRAG，收集所有运行信息，打包成一条结果记录返回。

    网络抖动是常态（国内 + 代理，一轮要调上百次 LLM），所以单题重试。
    **重试耗尽不抛异常**，返回一条 grade="error" 的占位记录让整轮继续 ——
    否则一次连接错误就废掉 20-40 分钟的一整轮。
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            # 合并初始状态，注入当前问题
            state = {**INITIAL_STATE, "question": case.question}
            result = app.invoke(state)
        except Exception as e:
            last_err = e
            print(f"[警告] {case.id} 第 {attempt}/{max_attempts} 次跑图失败：{type(e).__name__}: {e}")
            continue

        branch_actual = detect_branch(result, result.get("contexts", []))
        return {
            "id": case.id,
            "type": case.type,
            "question": case.question,
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "grade": result.get("grade", ""),
            "retry_count": result.get("retry_count", 0),
            "branch_actual": branch_actual,
            "branch_expected": case.expect_branch,
            "scores": {}
        }

    print(f"[警告] {case.id} 重试 {max_attempts} 次仍失败，记为 error 并继续")
    return _failed_case(case, last_err)


def run_all(cases, app, llm=None, emb=None) -> dict:
    """TODO 3：跑全部题目，组装 run dict。
    1. run_cases = [run_case(c, app) for c in cases]
    2. 若 llm 不为 None：
         metrics = build_metrics(llm, emb)
         scored  = score(cases, run_cases, metrics)    # {case_id: {指标: 分}}
         把 scored 按 id 填回每题的 "scores"（没分保持 {}）
    3. git_commit：subprocess 跑 git rev-parse --short HEAD，失败填 "unknown"
    4. run_id：datetime.now().strftime("%Y-%m-%d-%H%M")
    5. 返回 {"run_id", "git_commit", "config", "cases": run_cases, "summary": summarize(run_cases)}
       config 先硬编码 {"reranker": "cross-encoder", "top_k": 3}
    """
    #`cases` 就是我们自己手动编写的评测题库
    # 1. 遍历所有样例，逐个跑CRAG图。
    # 打印进度并 flush —— 一轮 20-40 分钟，没有进度输出等于黑盒
    run_cases = []
    for i, c in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] 跑图：{c.id} ({c.type}) ...", flush=True)
        run_cases.append(run_case(c, app))

    # 2. llm不为None才执行RAGAS打分
    #执行完这一段，run_cases 里面每一条记录的`scores`就填充上 RAGAS 分数了
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

    # ========= (b) 返回字典新增 summary 字段 =========
    return {
        "run_id": run_id,
        "git_commit": git_commit,
        # 记录真实生效的配置 —— 硬编码会让两份 run 的对比失去意义
        "config": {"reranker": os.getenv("RERANKER", "cross"), "top_k": 3},
        "cases": run_cases,
        "summary": summarize(run_cases),
    }


def save_run(run: dict, runs_dir: Path = RUNS_DIR) -> Path:
    #接收 `run_all` 生成的**完整评测大字典**，把它持久化保存为 json 文件
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
    # ========= (c) 新增 --compare 参数 =========
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"),
                        help="对比两份 run 文件，不跑评测")
    args = parser.parse_args(argv)

    # ========= --compare 分支：直接对比，不跑评测 =========
    if args.compare:
        old = json.load(open(args.compare[0], encoding="utf-8"))
        new = json.load(open(args.compare[1], encoding="utf-8"))
        print(diff_runs(old, new))
        return

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
    # ========= 修改打印：先汇总表，再逐题明细 =========
    print(format_summary(run_result["summary"]))
    print_behavior_table(run_result)


if __name__ == "__main__":
    main()
