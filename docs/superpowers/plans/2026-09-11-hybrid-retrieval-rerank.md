# 混合检索 + Rerank 实现计划（阶段 2 · 子系统一）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **⚠️ 本项目采用「学习式开发」（见 spec §10）**：用户自己写实现。助手的职责是**按任务逐个交底**——给出精确签名、返回结构、边界分支、关键 API 片段、易错点，然后把核心逻辑留白给用户填。不要替用户把实现写完。
>
> **每个任务的开场格式：** 先贴该任务的「隔板（骨架）」+ 测试代码，说明每个 TODO 要干嘛，然后**停下来等用户写完**，再跑测试、排错、提交。测试代码可直接给出（测试是规格，不是学习目标）。

**Goal:** 把阶段 1 的纯向量检索升级为「向量 + BM25 双路粗筛 → RRF 融合 → 交叉编码器精排」三段式，且不破坏任何现有对外接口。

**Architecture:** 新增 `adaptive_search_rag.retrieval` 包（3 个模块），在 `nodes/retrieve.py` 的 `build_retriever()` 内部完成组装。`retrieve_node` 签名、`RAGState`、`contexts: list[str]` 结构全部不变——`graph`/`route`/`generate`/`search` 无感知。

**Tech Stack:** `rank-bm25` + `jieba`（中文分词）、`sentence-transformers` 6.0.1 的 `CrossEncoder`、LangChain `with_structured_output`（LLM 打分）、pytest。

**Spec:** `docs/superpowers/specs/2026-09-10-hybrid-retrieval-rerank-design.md`（已评审定稿，commit `2babe81`）

## Global Constraints

- Python `>=3.12,<3.14`；包名 `adaptive_search_rag`；测试 `pythonpath=["src"]`，`pytest` 默认带 `-v`。
- **对外接口冻结**：不改 `state.py` / `graph.py` / `nodes/route.py` / `nodes/generate.py` / `nodes/search.py` 的签名；不改 `retrieve_node(state, top_k=3, retriever=None)`；`contexts` 永远是 `list[str]`。
- **测试隔离铁律**：测试永不触碰 HF 下载、网络 API、磁盘向量库。真实 CrossEncoder / DeepSeek LLM / Chroma 只在 `build_retriever()` 内加载。
- 所有模型加载一律 `local_files_only=True`。
- 常量值（写死，勿改）：`VECTOR_K = 10`、`BM25_K = 10`、`CANDIDATE_K = 8`、`RRF_K = 60`。RRF 的 `rank` 从 **1** 起。
- Windows 环境：跑脚本前 `sys.stdout.reconfigure(encoding="utf-8")`；`python -c` 含中文容易折行，用脚本文件。
- 提交风格：延续阶段 1，一个组件一个 commit。

## File Structure

| 文件 | 职责 | 动作 |
|---|---|---|
| `src/adaptive_search_rag/retrieval/__init__.py` | 空包标记 | Create |
| `src/adaptive_search_rag/retrieval/bm25.py` | 中文分词 + BM25 索引与检索闭包 | Create |
| `src/adaptive_search_rag/retrieval/rerank.py` | 三个 reranker：CrossEncoder（默认）/ LLM / NoOp 降级 | Create |
| `src/adaptive_search_rag/retrieval/hybrid.py` | 常量 + `rrf_fuse` + `build_hybrid_retriever` 编排 | Create |
| `src/adaptive_search_rag/nodes/retrieve.py` | 组装：`build_vector_retriever()` + `build_retriever()`（加缓存） | Modify |
| `tests/test_bm25.py` / `test_rerank.py` / `test_rrf.py` / `test_hybrid.py` | 各组件单测 | Create |
| `tests/test_retrieve.py` | **不动**（回归验证） | — |
| `scripts/download_reranker.py` | 模型预热（已在 `2babe81` 提交） | Done |
| `scripts/demo_hybrid.py` | 真实对比演示（纯向量 vs 混合 vs 精排） | Create |
| `README.md` | 补混合检索章节 | Modify |

---

### Task 1: 依赖声明与模型预热

**Files:**
- Modify: `pyproject.toml`（`dependencies` 数组）
- Already created: `scripts/download_reranker.py`

**Interfaces:**
- Consumes: 无
- Produces: `jieba`、`rank_bm25` 在 venv 中可用；`BAAI/bge-reranker-v2-m3` 在本地 HF 缓存中

- [ ] **Step 1: 在 `pyproject.toml` 的 `dependencies` 里加两行**

在 `"pydantic-settings>=2.5",` 之后追加：

```toml
    "rank-bm25>=0.2.2",
    "jieba>=0.42",
```

（包已经装进 `.venv` 了，这一步只是把声明补上，让 `pip install -e .` 可复现。）

- [ ] **Step 2: 验证依赖可用**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe -c "import jieba, rank_bm25; print('jieba/rank_bm25 OK')"
```

Expected: 打印 `jieba/rank_bm25 OK`（jieba 首次运行会 dump 缓存，约 0.5s，正常）

- [ ] **Step 3: 确认 reranker 模型已下齐**

```bash
ls ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/blobs/
```

Expected: 有一个**不带** `.incomplete` 后缀的 `model.safetensors`（约 2.27GB）。

若下载未完成（仍有 `.incomplete`），补跑：

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe scripts/download_reranker.py
```

- [ ] **Step 4: 全量回归，确认 18 个测试仍绿**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe -m pytest -q 2>&1 | tail -5
```

Expected: `18 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: declare rank-bm25 and jieba dependencies"
```

---

### Task 2: 中文 BM25 检索

**Files:**
- Create: `src/adaptive_search_rag/retrieval/__init__.py`（空文件）
- Create: `src/adaptive_search_rag/retrieval/bm25.py`
- Test: `tests/test_bm25.py`

**Interfaces:**
- Consumes: 无（纯 jieba + rank_bm25）
- Produces:
  - `tokenize(text: str) -> list[str]`
  - `build_bm25_retriever(chunks: list[str]) -> Callable[[str, int], list[str]]`，返回的闭包签名 `(question: str, top_k: int = 10) -> list[str]`，返回**原始 chunk 文本**

- [ ] **Step 1: 给用户交底（骨架）**

告诉用户 `bm25.py` 要实现两个函数，附下面这段骨架和说明：

```python
"""中文 BM25 检索：jieba 分词 + rank_bm25。

BM25 是词频统计型检索（稀疏检索），和向量检索互补：
- 向量检索擅长语义相近，但对专有名词/缩写/编号不敏感
- BM25 擅长精确词命中，但不懂同义改写
两者融合即为工业标准的「混合检索」。
"""
import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """中文分词。jieba.cut 返回生成器，必须 list() 化。"""
    # TODO 1: 用 jieba.cut 切分 text，返回 list[str]
    pass


def build_bm25_retriever(chunks: list[str]):
    """对 chunks 建 BM25 索引，返回检索闭包。

    返回的闭包签名必须与 build_vector_retriever 的闭包完全同构：
        retriever(question: str, top_k: int = 10) -> list[str]
    返回值是【原始 chunk 文本】，不是分词结果、不是元组。
    """
    # TODO 2: 对每个 chunk 调用 tokenize，得到二维 token 列表 corpus（List[List[str]]）
    # TODO 3: bm25 = BM25Okapi(corpus) 建索引（注意：这里必须保存原始 chunks 的顺序）

    def retriever(question: str, top_k: int = 10) -> list[str]:
        # TODO 4: 用 tokenize(question) 得到查询词列表
        # TODO 5: scores = bm25.get_scores(query_tokens)   # numpy.ndarray，长度 == len(chunks)
        # TODO 6: 按分数降序取前 top_k 个【下标】，再映射回原始 chunks 文本返回
        #         提示：sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        pass

    return retriever
```

**交底时强调的易错点：**
1. `BM25Okapi(corpus)` 的语料是**分词后的二维列表**，但返回值必须是**原始 `chunks` 文本**——两者顺序一一对应，别把分词结果当结果返回。
2. 排序要取**下标**再映射，不能直接对 `scores` 排序（你需要知道分数对应的原始位置用的是哪个 chunk）。
3. `scores[i]` 是 numpy 标量，比较/排序前显式 `float()` 转换最稳妥。
4. `chunks` 为空列表时不能崩（返回 `[]`）。
5. 别在这个模块里 import chromadb 或任何模型——它是纯 CPU 词频检索。

- [ ] **Step 2: 写测试（可直接给用户）**

创建 `tests/test_bm25.py`：

```python
from adaptive_search_rag.retrieval.bm25 import build_bm25_retriever, tokenize

# 小而可控的语料：每个 chunk 有明确的主题词
CHUNKS = [
    "RAG 通过检索外部知识来减少大模型的幻觉问题",
    "LangGraph 用 StateGraph 定义节点和边，支持条件路由",
    "BM25 是一种基于词频的稀疏检索算法，擅长精确词匹配",
    "交叉编码器 CrossEncoder 会对 query 和文档做深度交互打分",
]


def test_tokenize_returns_list_of_str():
    tokens = tokenize("混合检索")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_bm25_exact_term_wins():
    """专有名词只出现在一个 chunk 里时，BM25 应把它排到第一。"""
    retriever = build_bm25_retriever(CHUNKS)
    result = retriever("CrossEncoder 是什么", top_k=1)
    assert len(result) == 1
    # 只有最后一个 chunk 含 "CrossEncoder" 这个精确词
    assert "CrossEncoder" in result[0]


def test_bm25_respects_top_k():
    retriever = build_bm25_retriever(CHUNKS)
    assert len(retriever("检索", top_k=2)) == 2


def test_bm25_returns_original_chunks_only():
    """返回的必须是原始 chunk 文本，不是分词后的碎片。"""
    retriever = build_bm25_retriever(CHUNKS)
    for doc in retriever("检索算法", top_k=3):
        assert doc in CHUNKS


def test_bm25_empty_corpus_does_not_crash():
    retriever = build_bm25_retriever([])
    assert retriever("任意问题", top_k=3) == []
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe -m pytest tests/test_bm25.py -v 2>&1 | tail -15
```

Expected: 5 个测试 FAIL/ERROR（`assert None == 1` 之类），因为 `bm25.py` 里 TODO 还是 `pass`。

- [ ] **Step 4: 用户写实现**

用户填完 TODO 1~6。助手在卡住时给最小片段，不直接给完整解。

写完后跑：

```bash
.venv/Scripts/python.exe -m pytest tests/test_bm25.py -v 2>&1 | tail -15
```

Expected: `5 passed`

- [ ] **Step 5: 全量回归**

```bash
.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -3
```

Expected: `23 passed`

- [ ] **Step 6: Commit**

```bash
git add src/adaptive_search_rag/retrieval/__init__.py src/adaptive_search_rag/retrieval/bm25.py tests/test_bm25.py
git commit -m "feat: add chinese bm25 retriever"
```

---

### Task 3: 三个 Reranker

**Files:**
- Create: `src/adaptive_search_rag/retrieval/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `adaptive_search_rag.llm.get_llm()`（阶段 1 已有，返回 `ChatOpenAI`）
- Produces（三个类共享同一签名，这是「可切换」的基石）：
  - `CrossEncoderReranker(model_name: str = "BAAI/bge-reranker-v2-m3", model=None).rerank(question: str, docs: list[str], top_k: int) -> list[str]`
  - `LLMReranker(llm=None).rerank(question: str, docs: list[str], top_k: int) -> list[str]`
  - `NoOpReranker().rerank(question: str, docs: list[str], top_k: int) -> list[str]`
  - `RelevanceScore`（TypedDict，字段 `score: float`）

- [ ] **Step 1: 给用户交底（骨架）**

```python
"""精排（rerank）：粗筛候选池 → 用交叉编码器或 LLM 重排。

为什么需要精排：
- 粗筛（向量/BM25）是 Bi-Encoder，query 和 doc 各自独立编码，没有交互
- Cross-Encoder 把 [query, doc] 拼成一对送进模型，能做深度交互
  准确度约高一个量级，但速度慢约 100 倍 → 所以只对 top-8 候选池做
"""
from typing import TypedDict

from adaptive_search_rag.llm import get_llm


class RelevanceScore(TypedDict):
    """LLM 打分器期望的结构化输出（与 route.py 的 RouteDecision 同构）。"""
    score: float


SCORE_PROMPT = """你是一个相关性评分器。给定用户问题和一段文档，输出 0 到 1 的相关性分数。
- 1.0 表示文档直接回答了问题
- 0.0 表示完全不相关
只输出分数，不要解释。"""


class CrossEncoderReranker:
    """默认精排器：BAAI/bge-reranker-v2-m3。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model=None):
        self.model_name = model_name
        self.model = model          # 注入口：测试传假 model

    def _get_model(self):
        # TODO 1: 懒加载——如果 self.model 还是 None，才真正 import 并加载：
        #   from sentence_transformers import CrossEncoder
        #   self.model = CrossEncoder(self.model_name, local_files_only=True)
        # 然后 return self.model
        pass

    def rerank(self, question: str, docs: list[str], top_k: int) -> list[str]:
        # TODO 2: docs 为空 → 直接返回 []
        # TODO 3: pairs = [[question, d] for d in docs]
        # TODO 4: scores = self._get_model().predict(pairs)   # numpy.ndarray，顺序与 pairs 对齐
        # TODO 5: 按 float(scores[i]) 降序取前 top_k 个【下标】，映射回 docs 返回
        pass


class LLMReranker:
    """可替换精排器：逐条让 LLM 打分。每个 doc 一次调用，简单、易测、可并行。"""

    def __init__(self, llm=None):
        # TODO 6: self.llm = (llm or get_llm()).with_structured_output(
        #             RelevanceScore, method="function_calling")
        # 注意：with_structured_output 返回新对象，不会改到共享的 get_llm() 实例
        pass

    def rerank(self, question: str, docs: list[str], top_k: int) -> list[str]:
        # TODO 7: 对每个 doc 算分：
        #   messages = [SystemMessage(content=SCORE_PROMPT),
        #               HumanMessage(content=f"问题：{question}\n\n文档：{doc}")]
        #   score = self.llm.invoke(messages)["score"]
        # 单条打分抛异常时，该 doc 记 0.0 分，不要让整体崩掉（与 search_node 降级风格一致）
        # TODO 8: 按分数降序取前 top_k 个【下标】，映射回 docs 返回
        pass


class NoOpReranker:
    """降级实现：CrossEncoder 加载失败时使用，不做精排，按原顺序截断。"""

    def rerank(self, question: str, docs: list[str], top_k: int) -> list[str]:
        # TODO 9: 直接返回 docs[:top_k]
        pass
```

**交底时强调的易错点：**
1. **`predict()` 返回 numpy ndarray**——排序要 `float(scores[i])` 转换。直接 `sorted(zip(scores, docs))` 在 numpy 标量上不可靠。
2. `CrossEncoderReranker.rerank` 里**不要**在 `__init__` 加载模型，必须懒加载到 `_get_model()`，否则测试构造对象时就会去碰 HF。
3. 三个类返回的字符串**必须是入参 `docs` 里的原对象**，不能改写、不能新造。
4. `LLMReranker` 的逐条打分要用 `enumerate` 保留原下标，否则排序后无法映射回文档。
5. 导入 `SystemMessage, HumanMessage` 时注意 `from langchain_core.messages import ...`。
6. 别在这个文件顶层 import `sentence_transformers`——那会让整个模块在没下模型时无法 import，破坏测试隔离。必须在 `_get_model()` 里局部 import。

- [ ] **Step 2: 写测试（可直接给用户）**

创建 `tests/test_rerank.py`：

```python
import numpy as np
import pytest

from adaptive_search_rag.retrieval.rerank import (
    CrossEncoderReranker,
    LLMReranker,
    NoOpReranker,
)

DOCS = ["文档A", "文档B", "文档C"]


class _FakeCrossEncoder:
    """假交叉编码器：故意返回 numpy.ndarray（真实 predict 就是 ndarray），
    这样才测得到 float() 转换那条路径。"""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return np.array(self.scores, dtype=np.float32)


def test_cross_encoder_rerank_sorts_by_score():
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])      # B 最高分
    reranker = CrossEncoderReranker(model=fake)

    result = reranker.rerank("任意问题", DOCS, top_k=2)

    assert result == ["文档B", "文档C"]
    # 断言确实把 [query, doc] 成对送进去了
    assert fake.calls[0] == [["任意问题", "文档A"], ["任意问题", "文档B"], ["任意问题", "文档C"]]


def test_cross_encoder_rerank_top_k_exceeds_docs():
    fake = _FakeCrossEncoder([0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(model=fake)
    assert len(reranker.rerank("q", DOCS, top_k=10)) == 3


def test_cross_encoder_empty_docs():
    fake = _FakeCrossEncoder([])
    reranker = CrossEncoderReranker(model=fake)
    assert reranker.rerank("q", [], top_k=3) == []


def test_cross_encoder_returns_original_docs():
    fake = _FakeCrossEncoder([0.3, 0.2, 0.1])
    reranker = CrossEncoderReranker(model=fake)
    for doc in reranker.rerank("q", DOCS, top_k=3):
        assert doc in DOCS


class _FakeLLM:
    def __init__(self, scores):
        self.scores = scores
        self.i = 0

    def invoke(self, messages):
        score = self.scores[self.i]
        self.i += 1
        if score == "boom":
            raise RuntimeError("模拟 API 崩溃")
        return {"score": score}


def test_llm_reranker_sorts_by_score():
    reranker = LLMReranker(llm=_FakeLLM([0.2, 0.8, 0.5]))
    assert reranker.rerank("q", DOCS, top_k=2) == ["文档B", "文档C"]


def test_llm_reranker_survives_single_call_failure():
    """单条打分崩溃时记 0 分，整体不崩。"""
    reranker = LLMReranker(llm=_FakeLLM([0.9, "boom", 0.5]))
    result = reranker.rerank("q", DOCS, top_k=3)
    assert result == ["文档A", "文档C", "文档B"]   # B 记 0 分排最后


def test_noop_reranker_keeps_order():
    assert NoOpReranker().rerank("q", DOCS, top_k=2) == ["文档A", "文档B"]
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe -m pytest tests/test_rerank.py -v 2>&1 | tail -15
```

Expected: 7 个 FAIL（`AttributeError: 'NoneType' object has no attribute ...'` 或断言不符）

- [ ] **Step 4: 用户写实现**

用户填完 TODO 1~9。跑：

```bash
.venv/Scripts/python.exe -m pytest tests/test_rerank.py -v 2>&1 | tail -15
```

Expected: `7 passed`

- [ ] **Step 5: 全量回归**

```bash
.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -3
```

Expected: `30 passed`

- [ ] **Step 6: Commit**

```bash
git add src/adaptive_search_rag/retrieval/rerank.py tests/test_rerank.py
git commit -m "feat: add cross-encoder and llm rerankers"
```

---

### Task 4: RRF 融合（纯函数，最适合热身）

**Files:**
- Create: `src/adaptive_search_rag/retrieval/hybrid.py`（本节只放常量和 `rrf_fuse`）
- Test: `tests/test_rrf.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - 常量 `VECTOR_K = 10`、`BM25_K = 10`、`CANDIDATE_K = 8`、`RRF_K = 60`
  - `rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]`

- [ ] **Step 1: 给用户交底（骨架）**

```python
"""混合检索编排：RRF 融合 + 三组件串起来。

RRF（Reciprocal Rank Fusion）解决的核心问题：
BM25 的分数是「词频得分」（可以到几十），向量检索的分数是「余弦相似度」（0~1），
量纲完全不同，不能直接加权相加。RRF 只看【排名】不看【分数绝对值】，
天然消除量纲差异，所以是工业标准做法。

公式：RRF(d) = Σ_i 1 / (k + rank_i(d))      k 取 60，rank 从 1 开始
"""
from collections import defaultdict

VECTOR_K = 10       # 向量粗筛召回数：粗筛要宽，避免漏
BM25_K = 10         # BM25 粗筛召回数：与向量对齐
CANDIDATE_K = 8     # 融合后的候选池大小：太小漏答案，太大 rerank 慢
RRF_K = 60          # RRF 公式里的 k，业界惯例


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """多路排名融合，返回降序排列的文档列表。

    rankings: 每一路是一个「已按相关性降序排好的文档列表」，如
              [["docA", "docB"], ["docB", "docC"]]
    返回:     融合后的文档列表（去重，按 RRF 分数降序）

    关键语义（易错，务必按这个写）：
    同一个文档出现在多路里时，它的分数是【多路贡献之和】——
    这正是 RRF 奖励「多路共识」的机制，不是 bug。
    """
    # TODO 1: 用 defaultdict(float) 累加每个文档的 RRF 分数
    # TODO 2: 遍历 rankings 里每一路，enumerate(ranking, start=1) 拿 rank（必须从 1 开始！）
    #         对每个文档累加 1.0 / (k + rank)
    # TODO 3: 返回按分数降序排列的 key 列表
    pass
```

**交底时强调的易错点：**
1. **`rank` 从 1 开始**（用 `enumerate(ranking, start=1)`）。用 0 会算错。
2. **跨路重复要累加**，不要写成「先出现先赢」或「独立计分」。测试里有一条专门验证这个。
3. 边界：`rankings=[]` → `[]`；`[[], []]` → `[]`；只有一路 → 返回该路原顺序。
4. 分数相同时（如都在各自第 1 名）顺序会有歧义 → 加个稳定的 tie-breaker：按 `(分数降序, 最佳名次升序)`. 测试里有一条覆盖它。
5. 不要写 `list(dict.fromkeys(...))` 去重——dict 的 key 天然已经去重了，多余的操作会打乱顺序。

- [ ] **Step 2: 写测试（可直接给用户）**

创建 `tests/test_rrf.py`：

```python
import pytest

from adaptive_search_rag.retrieval.hybrid import rrf_fuse


def test_rrf_accumulates_across_rankings():
    """同一文档出现在两路时，分数是两路贡献之和。"""
    rankings = [["A", "B", "C"], ["A", "C"]]
    result = rrf_fuse(rankings, k=60)
    # A: 第1路 rank1 + 第2路 rank1 → 2/61
    # C: 第1路 rank3 + 第2路 rank2 → 1/63 + 1/62
    # B: 第1路 rank2               → 1/62
    # 期望顺序：A, C, B
    assert result == ["A", "C", "B"]


def test_rrf_single_ranking_preserves_order():
    assert rrf_fuse([["X", "Y", "Z"]], k=60) == ["X", "Y", "Z"]


def test_rrf_empty_inputs():
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []


def test_rrf_one_side_empty():
    assert rrf_fuse([[], ["P", "Q"]], k=60) == ["P", "Q"]


def test_rrf_deduplicates():
    """两路完全相同的列表，结果里每个文档只出现一次。"""
    result = rrf_fuse([["A", "B"], ["A", "B"]], k=60)
    assert result == ["A", "B"]
    assert len(result) == len(set(result))


def test_rrf_custom_k_is_used():
    """k=1 时手算可验证：rank1 → 0.5, rank2 → 0.333..."""
    result = rrf_fuse([["A", "B"]], k=1)
    assert result == ["A", "B"]


def test_rrf_tie_is_deterministic():
    """两路里各自第 1 名的两个不同文档，分数相同，顺序必须稳定。"""
    r1 = rrf_fuse([["A"], ["B"]], k=60)
    r2 = rrf_fuse([["A"], ["B"]], k=60)
    assert r1 == r2
    assert set(r1) == {"A", "B"}
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe -m pytest tests/test_rrf.py -v 2>&1 | tail -15
```

Expected: 7 个 FAIL（返回 `None`）

- [ ] **Step 4: 用户写实现**

用户填完 TODO 1~3（约 5 行）。跑：

```bash
.venv/Scripts/python.exe -m pytest tests/test_rrf.py -v 2>&1 | tail -15
```

Expected: `7 passed`

- [ ] **Step 5: 全量回归**

```bash
.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -3
```

Expected: `37 passed`

- [ ] **Step 6: Commit**

```bash
git add src/adaptive_search_rag/retrieval/hybrid.py tests/test_rrf.py
git commit -m "feat: add rrf fusion"
```

---

### Task 5: 混合检索编排

**Files:**
- Modify: `src/adaptive_search_rag/retrieval/hybrid.py`（追加 `build_hybrid_retriever`）
- Test: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: Task 4 的常量与 `rrf_fuse`
- Produces: `build_hybrid_retriever(vector_retriever, bm25_retriever, reranker) -> Callable[[str, int], list[str]]`，返回的闭包签名 `(question: str, top_k: int = 3) -> list[str]`

- [ ] **Step 1: 给用户交底（骨架）**

```python
def build_hybrid_retriever(vector_retriever, bm25_retriever, reranker):
    """三组件编排。三个都是必填参数，由调用方（retrieve.py 的 build_retriever）组装传入——
    hybrid 只做编排，不负责建真实组件（它不知道 Chroma 配置、也不知道模型名）。

    vector_retriever / bm25_retriever : (question, top_k) -> list[str]
    reranker                          : .rerank(question, docs, top_k) -> list[str]
    """

    def retriever(question: str, top_k: int = 3) -> list[str]:
        # TODO 1: rankings = [vector_retriever(question, VECTOR_K),
        #                     bm25_retriever(question, BM25_K)]
        # TODO 2: fused = rrf_fuse(rankings, RRF_K)
        # TODO 3: candidates = fused[:CANDIDATE_K]
        # TODO 4: return reranker.rerank(question, candidates, top_k)
        #         ↑ 注意最后用入参 top_k，不是 CANDIDATE_K
        pass

    return retriever
```

**交底时强调的易错点（这条最重要）：**
**`top_k` 的传递链**——`retrieve_node(top_k=3)` → 混合闭包的 `top_k` → 直接透传给 `reranker.rerank` 作为**最终条数**。粗筛用的 `VECTOR_K`/`BM25_K`/`CANDIDATE_K` 是内部常量，跟 `top_k` 无关。搞混了就会出现「要 3 条给出 8 条」。
- `candidates` 为空时 `reranker.rerank(question, [], top_k)` 应该返回 `[]`（Task 3 已经保证），hybrid 不用额外判断。
- 参数名用 `bm25_retriever` 而不是 `bm25`，避免和模块名 `bm25.py` 混淆。

- [ ] **Step 2: 写测试（可直接给用户）**

创建 `tests/test_hybrid.py`：

```python
from adaptive_search_rag.retrieval.hybrid import (
    BM25_K,
    CANDIDATE_K,
    VECTOR_K,
    build_hybrid_retriever,
)


class _RecordingRetriever:
    """记录被调用参数的假粗筛器，返回固定排名。"""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, question, top_k):
        self.calls.append((question, top_k))
        return self.result[:top_k]


class _RecordingReranker:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def rerank(self, question, docs, top_k):
        self.calls.append((question, list(docs), top_k))
        if self.result is not None:
            return self.result
        return docs[:top_k]


def test_hybrid_orchestration_order_and_kwargs():
    vector = _RecordingRetriever(["V1", "V2", "V3", "V4", "V5"])
    bm25 = _RecordingRetriever(["B1", "B2"])
    reranker = _RecordingReranker()

    retriever = build_hybrid_retriever(vector, bm25, reranker)
    result = retriever("问题", top_k=2)

    # 1. 两路粗筛各自用了正确的 K
    assert vector.calls == [("问题", VECTOR_K)]
    assert bm25.calls == [("问题", BM25_K)]

    # 2. 送进 rerank 的是 RRF 融合后截断的候选池（不超过 CANDIDATE_K）
    q, docs, k = reranker.calls[0]
    assert q == "问题"
    assert len(docs) <= CANDIDATE_K
    assert set(docs) == {"V1", "V2", "V3", "V4", "V5", "B1", "B2"}

    # 3. 最终条数由 top_k 决定
    assert k == 2
    assert result == docs[:2]


def test_hybrid_uses_reranker_output_verbatim():
    """hybrid 不加工 reranker 的返回值。"""
    vector = _RecordingRetriever(["V1", "V2"])
    bm25 = _RecordingRetriever(["B1"])
    reranker = _RecordingReranker(result=["重排后的唯一结果"])

    result = build_hybrid_retriever(vector, bm25, reranker)("q", top_k=3)
    assert result == ["重排后的唯一结果"]


def test_hybrid_both_sides_empty():
    vector = _RecordingRetriever([])
    bm25 = _RecordingRetriever([])
    reranker = _RecordingReranker()

    result = build_hybrid_retriever(vector, bm25, reranker)("q", top_k=3)
    assert result == []
    assert reranker.calls[0][1] == []      # 送进 rerank 的是空候选池


def test_hybrid_default_top_k_is_three():
    vector = _RecordingRetriever([f"V{i}" for i in range(10)])
    bm25 = _RecordingRetriever([f"B{i}" for i in range(10)])
    reranker = _RecordingReranker()

    build_hybrid_retriever(vector, bm25, reranker)("q")
    assert reranker.calls[0][2] == 3
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe -m pytest tests/test_hybrid.py -v 2>&1 | tail -15
```

Expected: `ImportError: cannot import name 'build_hybrid_retriever'`

- [ ] **Step 4: 用户写实现**

用户填完 TODO 1~4。跑：

```bash
.venv/Scripts/python.exe -m pytest tests/test_hybrid.py -v 2>&1 | tail -15
```

Expected: `4 passed`

- [ ] **Step 5: 全量回归**

```bash
.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -3
```

Expected: `41 passed`

- [ ] **Step 6: Commit**

```bash
git add src/adaptive_search_rag/retrieval/hybrid.py tests/test_hybrid.py
git commit -m "feat: add hybrid retrieval orchestration"
```

---

### Task 6: 接线进 `build_retriever`（真实跑通）

**Files:**
- Modify: `src/adaptive_search_rag/nodes/retrieve.py`
- Test: `tests/test_retrieve.py`（**不改**，仅回归）

**Interfaces:**
- Consumes: `build_bm25_retriever`（Task 2）、`CrossEncoderReranker` / `NoOpReranker`（Task 3）、`build_hybrid_retriever`（Task 5）
- Produces: `build_retriever()`（带 `lru_cache`，返回混合检索闭包）；`retrieve_node` 签名**不变**

- [ ] **Step 1: 给用户交底（骨架）**

告诉用户这次是**重构**：把现有 `build_retriever()` 里的向量闭包**原样搬**进一个新的 `build_vector_retriever()`，再新写一个组装的 `build_retriever()`。

```python
from functools import lru_cache

from adaptive_search_rag.retrieval.bm25 import build_bm25_retriever
from adaptive_search_rag.retrieval.hybrid import build_hybrid_retriever
from adaptive_search_rag.retrieval.rerank import CrossEncoderReranker, NoOpReranker
from adaptive_search_rag.state import RAGState


def build_vector_retriever():
    """建 Chroma collection，返回纯向量检索闭包（阶段的现有逻辑，原样搬过来）。"""
    # ===== 硬编码配置，务必和 ingest.py 里那三行完全一致 =====
    CHROMA_DB_PATH = "./chroma_db"
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    COLLECTION_NAME = "adaptive_knowledge"
    # ========================================================
    # TODO 1: 把现有 build_retriever() 里的代码整体搬到这里
    #         （import chromadb / embedding_functions、建 ef、PersistentClient、get_collection，
    #          以及内部 retriever(question, top_k=3) 闭包）
    #         最后 return retriever
    pass


@lru_cache(maxsize=1)
def build_retriever():
    """组装混合检索：向量 + BM25 双路粗筛 → RRF 融合 → CrossEncoder 精排。

    @lru_cache(maxsize=1)：进程内只构建一次（否则 CLI 每问一次都要重载
    Embedding + CrossEncoder 模型，慢到不可用）。
    长驻进程若更新了语料，需调用 build_retriever.cache_clear()。
    """
    # TODO 2: vector_retriever = build_vector_retriever()   # 建集合 + 向量闭包
    # TODO 3: 拿全量语料建 BM25。这里需要自己取一次集合（配置和 build_vector_retriever 里那三行一致）：
    #            client = chromadb.PersistentClient(path="./chroma_db")
    #            collection = client.get_collection("adaptive_knowledge")
    #            chunks = collection.get()["documents"]
    #              注意 get() 与 query() 的区别：get() 是全量拉取，返回一维 list；
    #              query() 才是向量相似度检索，返回 res["documents"][0]。
    #              （实测 chromadb 1.5.9：15 个 chunk 未传 limit 也全部返回。
    #                大语料需分页拉取或改落盘索引 —— 在注释里写明）
    #         ★ 若嫌重复，可以把「建 client+collection」抽成一个模块私有小函数
    #           _get_collection()，供 build_vector_retriever 和这里共用。
    # TODO 4: bm25_retriever = build_bm25_retriever(chunks)
    # TODO 5: reranker = _build_reranker()
    # TODO 6: return build_hybrid_retriever(vector_retriever, bm25_retriever, reranker)
    pass


def _build_reranker():
    """加载 CrossEncoder；失败则降级 NoOpReranker（保住 RRF 顺序，链路不断）。"""
    # TODO 7: 注意！CrossEncoderReranker 是【懒加载】的，构造它并不会真的加载模型。
    #         所以光 try 构造没用，必须显式触发一次真实加载来探活：
    #             try:
    #                 reranker = CrossEncoderReranker()
    #                 reranker._get_model()        # ← 真正加载模型，成功则返回它
    #                 return reranker
    #             except Exception as e:
    #                 print(f"[警告] CrossEncoder 加载失败，已降级为 NoOpReranker：{e}")
    #                 return NoOpReranker()
    #         （这就是 spec §5.2「降级策略」的落地；日志有警告 = 验收标准没过）
    pass


def retrieve_node(state: RAGState, top_k: int = 3, retriever=None) -> dict:
    """检索节点：从混合检索取 top_k 片段，返回统一 {"contexts": [...]}"""
    # 这个函数【完全不动】，保持现状
    ...
```

**交底时强调的易错点：**
1. **`retrieve_node` 一个字都不改**——这是 spec §7 的核心卖点，签名和返回值必须与阶段 1 一致。
2. `build_vector_retriever()` 里那段配置注释（`务必和 ingest.py 完全一致`）要搬过去一起保留。
3. `lru_cache` 要求函数可哈希参数——`build_retriever()` 零参，正好。**不要给它加参数**。
4. `collection.get()` 默认上限在 chromadb 新版是 10 条；实测本环境 1.5.9 下 15 个 chunk 未传 limit 也全部返回。仍建议显式 `limit=` 或在注释里写明「大语料需分页」。
5. `get()` 返回 `res["documents"]` 直接是一维 `list[str]`；`query()` 才是 `res["documents"][0]`。两者容易混。
6. **降级分支必须有**：`_build_reranker()` 里 try/except 包住 `CrossEncoderReranker()`。注意懒加载设计下，`CrossEncoderReranker()` 构造时**不会**加载模型——所以这里要显式触发一次加载来验证可用性（例如 `reranker._get_model()`），或者干脆把 try 放在首次调用时。**若想两者兼顾，可在 `_build_reranker()` 里 try 调用一次 `reranker._get_model()` 来提前探活。**
7. `build_retriever` 上方要保留「为什么缓存」的注释，这是面试讲点。

- [ ] **Step 2: 跑现有测试确认没破坏（回归）**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe -m pytest tests/test_retrieve.py tests/test_graph.py -v 2>&1 | tail -10
```

Expected: 全 PASS（`test_retrieve.py` 注入 fake，压根不碰新代码）

- [ ] **Step 3: 写一个真实探活脚本验证端到端**

创建临时验证（写进 `scripts/`，别用 `python -c`，Windows 下中文会折行）：

```python
# scripts/_smoke_hybrid.py
import sys

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.nodes.retrieve import build_retriever
from adaptive_search_rag.retrieval.hybrid import CANDIDATE_K, VECTOR_K, BM25_K


def main() -> None:
    retriever = build_retriever()
    q = "什么是 RAG 的幻觉问题"
    docs = retriever(q, top_k=3)
    print(f"常量: VECTOR_K={VECTOR_K} BM25_K={BM25_K} CANDIDATE_K={CANDIDATE_K}")
    print(f"返回条数: {len(docs)}")
    for i, d in enumerate(docs, 1):
        print(f"--- top{i} ---")
        print(d[:120])


if __name__ == "__main__":
    main()
```

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe scripts/_smoke_hybrid.py 2>&1 | tail -25
```

Expected: 打印 3 条结果，**且日志中不出现降级警告**（出现即说明 CrossEncoder 没加载成功，需排查模型路径）；首次运行会加载两个模型，耗时较长属正常。

验证完删除该临时脚本。

- [ ] **Step 4: CLI 端到端验证**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && echo "什么是 LangGraph 的条件路由
exit" | .venv/Scripts/python.exe -m adaptive_search_rag.main 2>&1 | tail -20
```

Expected: 能正常返回答案（说明 `route → retrieve(混合) → generate` 全链路通畅），无异常堆栈。

- [ ] **Step 5: 全量回归**

```bash
.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -3
```

Expected: `41 passed`（原 18 + 新增 23）

- [ ] **Step 6: Commit**

```bash
git add src/adaptive_search_rag/nodes/retrieve.py
git commit -m "refactor: wire hybrid retrieval into build_retriever"
```

---

### Task 7: 演示脚本与文档收尾

**Files:**
- Create: `scripts/demo_hybrid.py`
- Modify: `README.md`
- Modify: `notebook.txt`（用户自己补 Task7~11 的笔记，助手给要点提示）

**Interfaces:**
- Consumes: Task 2~6 全部
- Produces: 可复现的面试演示 + 更新的 README

- [ ] **Step 1: 写对比演示脚本**

创建 `scripts/demo_hybrid.py`：打印三组对照，用来在面试里现场演示「混合检索 + 精排」的价值。

```python
"""真实对比演示：纯向量检索 vs BM25 vs 混合+精排。

用法：.venv/Scripts/python.exe scripts/demo_hybrid.py
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from adaptive_search_rag.nodes.retrieve import build_vector_retriever
from adaptive_search_rag.retrieval.bm25 import build_bm25_retriever
from adaptive_search_rag.retrieval.hybrid import rrf_fuse, VECTOR_K, BM25_K
from adaptive_search_rag.nodes.retrieve import build_retriever
import chromadb
from chromadb.utils import embedding_functions

# 选一个「精确术语」问题，纯向量检索容易排不到位，BM25 能一击命中
QUESTIONS = [
    "RFC 7231",                       # 若语料含编号类术语
    "CrossEncoder 和 Bi-Encoder 有什么区别",
    "什么是 RAG 的幻觉问题",
]


def _load_chunks() -> list[str]:
    """从 chroma 全量取语料（用于建 BM25 索引）。"""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-zh-v1.5", local_files_only=True
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="adaptive_knowledge", embedding_function=ef)
    return collection.get()["documents"]


def main() -> None:
    # TODO 1: 造三个检索器
    #   vector = build_vector_retriever()
    #   bm25 = build_bm25_retriever(_load_chunks())
    #   hybrid = build_retriever()             # 已带 lru_cache，不会重复加载模型
    #
    # TODO 2: 对每个问题打印三路 top-3 摘要 + RRF 融合顺序
    #   输出格式建议：
    #     ===== 问题：xxx =====
    #     [向量]      1. <前60字>   2. ...   3. ...
    #     [BM25]      1. ...        2. ...   3. ...
    #     [混合+精排] 1. ...        2. ...   3. ...
    pass


if __name__ == "__main__":
    main()
```

**注意**：演示脚本会真实加载模型，只用于手动运行和面试演示，**不纳入 pytest 收集范围**（`testpaths = ["tests"]` 已保证这点）。

- [ ] **Step 2: 跑演示脚本，确认三路输出有可见差异**

```bash
cd /e/Ai/agent/AdaptiveSearch-RAG && .venv/Scripts/python.exe scripts/demo_hybrid.py 2>&1 | tail -40
```

Expected: 同一问题下，三路排序不同；至少有一个问题能看出 BM25 把含精确词的 chunk 提到了向量检索前面。**把这个对比结果记下来，是面试的现场素材。**

- [ ] **Step 3: 更新 README**

在 README 的检索相关章节补充（按现有 README 风格写）：

- 一句架构：`向量粗筛(top-10) + BM25 粗筛(top-10) → RRF 融合(top-8) → CrossEncoder 精排(top-3)`
- RRF 公式与「为什么不用加权求和」（量纲不同）
- 三个可切换的 reranker：`CrossEncoderReranker`（默认）/ `LLMReranker` / `NoOpReranker`（降级）
- 新增依赖与模型预热说明（指向 `scripts/download_reranker.py`）
- 演示命令：`.venv/Scripts/python.exe scripts/demo_hybrid.py`

- [ ] **Step 4: 全量回归**

```bash
.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -3
```

Expected: `41 passed`

- [ ] **Step 5: 更新 notebook.txt（提示用户自己补）**

给用户列这几条要点，让他自己写进笔记：
- 混合检索为什么是工业标准（稀疏 + 密集互补）
- RRF 公式、rank 从 1 起、跨路累加、k=60 的由来
- Cross-Encoder vs Bi-Encoder 的速度/精度 trade-off
- `CANDIDATE_K` 的取值权衡（太小漏答案、太大 rerank 慢）
- 依赖注入在三个组件上的体现（测试全 fake）
- `lru_cache` 为什么必须加
- Xet 401 那个坑

- [ ] **Step 6: Commit 并推送**

```bash
git add scripts/demo_hybrid.py README.md
git commit -m "docs: add hybrid retrieval demo and README"
git push origin master
```

（push 需要代理，若 Connection reset 提醒用户开工具。）

---

## 验收对照（spec §11）

| 验收标准 | 对应任务 |
|---|---|
| `pytest -v` 全绿，新增 4 个测试文件 + 原 18 不回归 | Task 1~7 每步的全量回归 |
| `build_retriever()` 真实加载 CrossEncoder（不走降级） | Task 6 Step 3 探活脚本 |
| CLI 真实跑通混合检索 | Task 6 Step 4 |
| 能演示 BM25 精确词命中 vs 纯向量 | Task 7 Step 2 |
| 能默写 RRF 公式、讲清 trade-off | Task 4 / Task 7 Step 5 |
| 交叉编码器 vs LLM-as-reranker 可切换 | Task 3 |
