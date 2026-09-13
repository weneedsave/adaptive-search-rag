from __future__ import annotations
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)
from evaluation.golden import QUALITY_TYPES

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
#分别对应:事实一致性,回答相关性,上下文召回,上下文精确度

def build_judge_llm():
    """
    client =
    AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL",
    "https://api.deepseek.com"))
    return llm_factory(os.getenv("DEEPSEEK_MODEL",
    "deepseek-v4-flash"), client=client)
    用 AsyncOpenAI 而非 OpenAI：ascore 是 async
    的（spec §8.2）。
    """
    load_dotenv()
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    # 两个参数都不能省（均为 spec §8.6 实测结论）：
    #   max_tokens：真实答案的 statement 提取会输出一长串原子命题，
    #              默认上限和中等的 8192 都偶发撑爆，报 IncompleteOutputException
    #   temperature：DeepSeek 是 MoE，temperature=0 **也不能完全消除**抖动，
    #              但方向正确，且与 llm.py 的生产配置保持一致
    # ⚠️ 裁判分数有 ±0.03 量级的固有噪声，小于 ~0.05 的差异不可当信号解读
    return llm_factory(
        os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        client=client,
        max_tokens=16384,
        temperature=0,
    )


def build_embeddings():
    """
    normalize_embeddings=True, batch_size=32)
    注意大写 F —— 要现代版（spec §8.3）。
    """
    return HuggingFaceEmbeddings(
        model="BAAI/bge-small-zh-v1.5",
        normalize_embeddings=True,
        batch_size=32
    )


def build_metrics(llm, emb) -> dict[str, object]:
    """
    METRIC_NAMES 里的字符串。
    四个构造函数参数**不一样**（spec §8.5）：
      Faithfulness(llm=llm)
      AnswerRelevancy(llm=llm, embeddings=emb)
    ← 唯一要 embeddings 的
      ContextRecall(llm=llm)
      ContextPrecisionWithoutReference(llm=llm)
    """
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),#回答相关性,需要检索上下文和用户问题向量相似度所以需要emb
        "context_recall": ContextRecall(llm=llm),
        "context_precision": ContextPrecisionWithoutReference(llm=llm)
    }


def select_pairs(cases, run_cases) -> list[tuple]:
    """
    只保留 type ∈ QUALITY_TYPES 且 id 能在
    run_cases 里找到的题。
    保持 cases 的原顺序（测试会检查顺序）。
    """
    pairs = []
    run_map = {rc["id"]: rc for rc in run_cases}
    for case in cases:
        #QUALITY_TYPES = {"kb", "kb-hard", "rewrite"}
        if case.type in QUALITY_TYPES and case.id in run_map:
            pairs.append((case, run_map[case.id]))
    return pairs


def metric_args(name: str, case, run_case) -> dict:
    """
    ⚠️ 参数名写错 =
    指标静默算错，不报错。逐条对照：
      faithfulness       user_input, response,
    retrieved_contexts
      answer_relevancy   user_input, response
      context_recall     user_input,
    retrieved_contexts, reference
      context_precision  user_input, response,
    retrieved_contexts
    映射来源：user_input <- case.question
             response   <- run_case["answer"]
             retrieved_contexts <-
    run_case["contexts"]
             reference  <- case.reference
    未知 name 抛 KeyError。
    """
    user_input = case.question  # 用户原始问题，来自golden样例
    response = run_case["answer"]  # CRAG大模型生成出来的回答
    retrieved_contexts = run_case["contexts"]  # 检索拿到的上下文chunk列表
    reference = case.reference  # golden里的标准答案（参考文本）

    if name == "faithfulness":
        return {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts
        }
    elif name == "answer_relevancy":
        return {
            "user_input": user_input,
            "response": response
        }
    elif name == "context_recall":
        return {
            "user_input": user_input,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference
        }
    elif name == "context_precision":
        return {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts
        }
    else:
        raise KeyError(f"未知指标名称: {name}")


async def _score_all(pairs, metrics) -> dict[str, dict]:
    """
    1. **`pairs`**select_pairs`输出的列表，格式：[(case, run_case), (case2, run_case2), ...]
case：GoldenCase 标准答案样例（question、reference、id、type）
run_case：CRAG 图跑完得到的结果字典（id、answer、contexts）
`metrics` 是 `build_metrics(llm, emb)` 返回的**字典**：
    """
    result_out = {}  # 最终结果容器，存放所有题的分数
    # =========外层循环：遍历每一道待评测题目=========
    for case, run_case in pairs:
        case_id = case.id  # 当前题目的id，比如kb-01
        score_dict = {}  # 保存这一道题的4个指标分数

        # =========内层循环：遍历4个评测指标=========
        for metric_name, metric_obj in metrics.items():
            # 1. metric_args：根据指标名字，拼装这个指标需要的参数字典args
            args = metric_args(metric_name, case, run_case)
            # 2. await metric_obj.ascore(**args)：调用裁判LLM打分，等待返回结果
            # 单题单指标失败不该毁掉整轮（一轮十几分钟），失败记 None 并出声告警。
            # 不静默吞掉：None 会在 report.summarize 里被跳过，不会当成 0 混进均值。
            try:
                metric_result = await metric_obj.ascore(**args)
            except Exception as e:
                print(f"[警告] {case_id} / {metric_name} 打分失败：{type(e).__name__}: {e}")
                score_dict[metric_name] = None
                continue
            # 3. metric_result是对象，metric_result.value取出0~1分数，存入当前题的分数字典
            score_dict[metric_name] = float(metric_result.value)

        # 当前题4个指标全部打完分，放进总结果
        result_out[case_id] = score_dict
    # 所有题目全部跑完，返回汇总分数
    return result_out
"""
result_out = {
    "kb-01": {                     # case_id
        "faithfulness": 0.82,      # score_dict里存的分数
        "answer_relevancy": 0.91,
        "context_recall": 0.75,
        "context_precision": 0.88
    }
}
"""
def score(cases, run_cases, metrics) -> dict[str, dict]:

    pairs = select_pairs(cases, run_cases)
    return asyncio.run(_score_all(pairs, metrics))
# `asyncio.run()` 的作用：**启动异步事件循环，执行异步函数 `_score_all`**