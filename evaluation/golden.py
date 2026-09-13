from __future__ import annotations
import json
#冻结对象，实例不可修改
from dataclasses import dataclass
from pathlib import Path

BLOCK_TYPES = ("kb", "kb-hard", "oob", "rewrite")
QUALITY_TYPES = ("kb", "kb-hard", "rewrite")#oob不参与打分,因为他完全是联网查询
BRANCHES = ("generate", "search", "rewrite")
MUST_REFUSE = "根据现有资料无法回答" #oob标准答案


@dataclass(frozen=True)
class GoldenCase:
    id: str
    type: str   #样本类型：kb /kb-hard/rewrite /oob
    question: str
    reference: str  #参考答案
    expect_branch: str | None = None #测试用例的期望分支
    note: str = "" #备注

#取出jsonl文件内容到一个列表中
def load_golden(path: str | Path) -> list[GoldenCase]:
    """逐行读 JSONL，返回 GoldenCase 列表。
    TODO：用 utf-8 打开，逐行 strip() 后跳过空行；
    用 json.loads 解析，构造 GoldenCase。
    expect_branch 用 row.get("expect_branch")（缺失时得到
    None，正是想要的默认）。
    note 同理用 row.get("note", "")。
    """
    p = Path(path)
    #读取并封装为一个字符串列表
    lines = p.read_text(encoding="utf-8").splitlines()
    cases: list[GoldenCase] = [] #创建列表,列表元素为 GoldenCase
    for line in lines:
        line = line.strip()#去除首尾空行
        if not line:
            continue
        row = json.loads(line)
        case = GoldenCase(
            id=row["id"],
            type=row["type"],
            question=row["question"],
            reference=row["reference"],
            expect_branch=row.get("expect_branch"),
            note=row.get("note", "")
        )
        cases.append(case)
    return cases


def validate_golden(cases: list[GoldenCase]) -> list[str]:
    """返回错误描述列表，空列表=通过。
    TODO：逐条检查并**收集**错误（不要 early return，一次报全）。
    每条错误信息里必须出现下面括号里的英文/中文词，
    测试会检查：
      1. id 为空                    → 信息含 "id"
      2. id 重复                    → 信息含 "重复"（需要先统计一遍所有 id）
      3. type 不在 BLOCK_TYPES      → 信息含 "type"
      4. question 为空              → 信息含 "question"
      5. reference 为空             → 信息含 "reference"
      6. expect_branch 既不是 None 也不在 BRANCHES → 信息含 "expect_branch"
      7. type == "oob" 且 reference != MUST_REFUSE → 信息含 "oob"
      错误信息里带上 case.id 会好排查很多。
    """
    errors: list[str] = []
    seen_ids: set[str] = set()

    for case in cases:
        # 1. id 为空
        if not isinstance(case.id, str) or len(case.id.strip()) == 0:
            errors.append(f"id 为空，case.id={case.id}")
        else:
            # 2. id 重复
            if case.id in seen_ids:
                errors.append(f"id 存在重复：{case.id}")
            seen_ids.add(case.id)

        # 3. type 不在 BLOCK_TYPES
        if case.type not in BLOCK_TYPES:
            errors.append(f"case {case.id}: 非法 type：{case.type}")

        # 4. question 为空
        if not isinstance(case.question, str) or len(case.question.strip()) == 0:
            errors.append(f"case {case.id}: question 不能为空")

        # 5. reference 为空
        if not isinstance(case.reference, str) or len(case.reference.strip()) == 0:
            errors.append(f"case {case.id}: reference 不能为空")

        # 6. expect_branch 校验
        if case.expect_branch is not None and case.expect_branch not in BRANCHES:
            errors.append(f"case {case.id}: 非法 expect_branch：{case.expect_branch}")

        # 7. oob 强制 reference == MUST_REFUSE
        if case.type == "oob":
            if case.reference != MUST_REFUSE:
                errors.append(f"case {case.id}: oob 类型样本 reference 必须为 MUST_REFUSE")

    return errors
