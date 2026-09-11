# 混合检索 + Rerank 设计文档（阶段 2 · 子系统一）

- 日期：2026-09-10（2026-09-11 评审修订）
- 状态：**待用户确认**（决策点已全部关闭，见各节「已定」标注）
- 定位：阶段 1「先路由式 Agentic RAG」的检索层升级，属于阶段 2 的第一个子系统
- 前置：阶段 1 已收官（18 测试全绿），本设计只动检索层，不改图结构、state、路由、生成

## 1. 背景与目标

阶段 1 的检索是**纯向量检索**（ChromaDB + bge-small-zh-v1.5 余弦相似度）。这是 Naive/Advanced RAG 的基线，但纯向量检索有三个致命缺陷（2026 RAG 面试高频必考）：

1. **精确匹配不行**——搜「RFC 7231」可能返回语义相关但不含精确术语的文档
2. **专业术语召回差**——缩写、产品名、专有名词易被语义相近但内容不符的 chunk 顶掉
3. **排序粗糙**——Bi-Encoder 独立编码 query/doc，无交互，召回的相关文档排不到前面

本设计把检索升级为工业标准的**「混合检索 + 精排」**三段式：

```
向量粗筛（语义） + BM25 粗筛（精确词） → RRF 融合 → 交叉编码器精排
```

**目标**：检索质量从「能跑」提升到「面试能讲 trade-off 的工业级」，且**不破坏阶段 1 的任何对外接口**。

## 2. 市场依据（为什么按这个顺序做）

2026 RAG 岗位调研结论：

- **混合检索 + Rerank = 几乎每轮必考，追问到算法层**（RRF 公式、BM25 vs 向量 trade-off）
- 交叉编码器是 rerank 的**标准答案**；LLM-as-reranker 是「会了标准项之后」的加分项
- 因此：主线用 bge 交叉编码器（标准、稳），注入接口留 `llm_reranker` 作可替换实现（前沿谈资）

## 3. 范围（做什么 / 不做什么）

**做：**
- 中文 BM25 检索（`rank_bm25` + `jieba` 分词）
- RRF 融合（自研 ~5 行，面试必默写）
- bge 交叉编码器精排（默认 reranker）
- LLM-as-reranker（可替换实现，复用现有 `get_llm()`）
- 三组件依赖注入 + 完整单测

**不做：**
- 不改 `state.py` / `graph.py` / `route.py` / `generate.py` / `search.py` / `nodes/retrieve.py` 的对外签名
- 不改 `contexts` 结构（仍是 `list[str]`）
- 不做 BM25 索引落盘（启动时从 Chroma 全量加载到内存）
- 本阶段不做统一加载层（PDF）、CRAG、RAGAS（后续子系统）

## 4. 数据流

```
question
  ├─→ 向量检索 top-10   (Chroma，语义召回)
  └─→ BM25 检索  top-10 (内存索引 + jieba，精确词召回)
          ↓ RRF 融合（k=60，量纲不同不能直接加权）
        候选池 top-8
          ↓ CrossEncoder 精排（query+doc 深度交互）
        top-3 → contexts
```

## 5. 组件设计

新增包 `src/adaptive_search_rag/retrieval/`，三个模块 + `__init__.py`：

```
src/adaptive_search_rag/retrieval/
├── __init__.py
├── bm25.py       # 中文 BM25（jieba 分词 + rank_bm25）
├── rerank.py     # CrossEncoderReranker（默认） + LLMReranker（可替换）
└── hybrid.py     # 混合检索编排 + RRF 融合
```

### 5.1 `bm25.py`

```python
import jieba
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    return list(jieba.cut(text))          # 中文必配分词，否则 BM25 按空格失效

def build_bm25_retriever(chunks: list[str]):
    """返回 bm25 检索闭包，签名 (question, top_k) -> list[str]，与 vector_retriever 对齐"""
    # 1. 对每个 chunk 分词，喂给 BM25Okapi 建索引
    # 2. 返回内部 retriever(question, top_k=10)：
    #    get_scores(tokenize(question)) → 按分数降序取 top_k → 映射回原始 chunks
    return retriever
```

**返回值约定（写死，别飘）：** 返回**原始 chunk 文本**的列表（不是分词结果、不是 `(chunk, score)` 元组），长度 `min(top_k, len(chunks))`，分数为 0 的 chunk 也允许出现在列表中（由下游 RRF/rerank 决定去留）。

**易错点：**
- BM25Okapi 建索引时要存**原始 chunk 顺序**，搜索时靠分数排名**映射回原 chunk**，别把分词后的 list 当返回结果。
- `jieba.cut` 返回生成器，要 `list()` 化。
- **bm25 组件必须是闭包 `(question, top_k) -> list[str]`**，和 `vector_retriever` 完全同构，这样 hybrid 层才能用同一个接口调用两路。
- 排序用 `sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_k]` 取**下标**再映射，不要直接对 `scores` 排序——你需要分数对应的原始下标。
- 实测 `jieba.cut("混合检索与重排序")` → `['混合', '检索', '与', '重', '排序']`。单字（如"重"）会被切出来，属正常。

### 5.2 `rerank.py`

```python
class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", model=None):
        # model 参数留注入口：测试传 fake，真实跑才 load CrossEncoder(local_files_only=True)
    def rerank(self, question: str, docs: list[str], top_k: int) -> list[str]:
        # pairs = [[question, d] for d in docs] → predict 打分 → 降序取 top_k

class LLMReranker:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
    def rerank(self, question, docs, top_k) -> list[str]:
        # 逐条打分：对每个 doc 让 LLM 输出相关性分数，降序取 top_k
```

**关键约定：** 两个 reranker 共用同一签名 `rerank(question, docs, top_k) -> list[str]`。hybrid 层只认这个接口，不关心实现——这是「可切换」的基石。

**返回值约定：** 与输入 `docs` 是**同一批字符串**（不能重新生成、不能改写文本），只是顺序变了并截断到 `top_k`；`docs` 为空时返回 `[]`。

**决策点：**
- `CrossEncoderReranker.__init__` 的 `model=None` 是注入口：真实跑才 `from sentence_transformers import CrossEncoder; CrossEncoder(model_name, local_files_only=True)`，测试传有 `predict()` 方法的假 model。
- **模型懒加载**：`__init__` 只存 `model_name` / 注入的 `model`，真正的 `CrossEncoder(...)` 推迟到首次 `rerank()` 里按需加载（`if self.model is None: ...`），这样测试构造对象时不会碰 HF。
- `LLMReranker` 用**逐条打分**（每个 doc 一次 LLM 调用，输出相关性分数），不用「一次性排序」——逐条更简单、可并行、易测试。打分 prompt 要求 LLM 只输出分数。
- 模型选 `BAAI/bge-reranker-v2-m3`（多语言，中文效果好于 `bge-reranker-base`）。

**实测 API 细节（sentence-transformers 6.0.1，别照抄旧教程）：**
- `CrossEncoder(model_name, local_files_only=True)` 可用；`model.predict([[query, doc], ...])` 返回 **`numpy.ndarray`**，顺序与输入 pairs 一一对应。
- **排序前必须 `float(score)` 转换**：直接 `sorted(zip(scores, docs), reverse=True)` 在 numpy 标量上行为不可靠，且 ndarray 是真值判断会炸。
  ```python
  pairs = [[question, d] for d in docs]
  scores = self.model.predict(pairs)          # numpy.ndarray
  order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
  return [docs[i] for i in order[:top_k]]
  ```
- `LLMReranker` 同样对接 `route_node` 的既有风格：`__init__` 里 `self.llm = (llm or get_llm()).with_structured_output(RelevanceScore, method="function_calling")`，`RelevanceScore` 定义为 `TypedDict`，只有 `score: float` 一个字段（和 `RouteDecision` 同构）。**注意 `with_structured_output` 返回的是新对象、不会改到共享的 `get_llm()` 实例。**
- LLM 打分兜底：单条打分调用抛异常时，该 doc 记 0 分而不是让整个 rerank 崩掉（与 `search_node` 的降级风格一致）。

**降级策略（已定）：** `build_retriever()` 里 try 加载真实 `CrossEncoder`，失败（模型没下齐 / OOM / 版本不兼容）则**打印警告并返回 `NoOpReranker`**——它不做任何精排，直接按 `docs` 原顺序（即 RRF 融合顺序）截 `top_k`。好处：混合检索本体仍能端到端跑通，不会因模型问题整条链路挂掉。`NoOpReranker` 同样实现 `rerank(question, docs, top_k) -> list[str]`，因此不需要额外测试分支。

### 5.3 `hybrid.py`

```python
def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    # RRF(d) = Σ 1/(k + rank_i(d))，多路排名融合，返回降序 list[str]
    # 空候选 / 单路 / 重复文档都要能正常返回

def build_hybrid_retriever(vector_retriever, bm25, reranker):
    """三组件必填，均由调用方（retrieve.py 的 build_retriever）组装传入。
    hybrid 只做编排，不负责建真实组件（它不知道 Chroma 配置）。"""
    def retriever(question: str, top_k: int = 3) -> list[str]:
        # 1. vector_retriever(question, VECTOR_K)   → rankings[0]
        # 2. bm25_retriever(question, BM25_K)       → rankings[1]
        # 3. candidate = rrf_fuse(rankings, RRF_K)[:CANDIDATE_K]
        # 4. return reranker.rerank(question, candidate, top_k)   ← top_k 是入参，不是 CANDIDATE_K
    return retriever
```

**`top_k` 的传递链（容易搞混，写死）：** `retrieve_node` 的 `top_k=3` → 混合检索闭包的参数 `top_k` → **直接透传给 `reranker.rerank` 作为最终返回条数**。粗筛用的 `VECTOR_K`/`BM25_K`/`CANDIDATE_K` 是内部常量，和 `top_k` 无关。即：`top_k` 控制「最终给 LLM 几条」，`CANDIDATE_K` 控制「送进精排几条」。

四个常量定义在 `hybrid.py` 模块顶部（**不要塞进闭包**，否则测试想改都改不了）：`VECTOR_K = 10`、`BM25_K = 10`、`CANDIDATE_K = 8`、`RRF_K = 60`。

**统一签名约定（关键）：** `vector_retriever`、`bm25` 都是 `(question, top_k) -> list[str]`；`reranker` 是 `rerank(question, docs, top_k) -> list[str]`。hybrid 层只认这些接口，测试时各传 fake 即可。

**`rrf_fuse` 实现决策（已定，写死）：**
- **按文档原文做 key 累加**：同一个 chunk 同时出现在两路排名里时，它的 RRF 分数是**两路贡献之和**——这正是 RRF 奖励「多路共识」的核心机制，不是 bug。「独立计分」的做法是错的。
- rank 从 **1** 开始计（不是 0），否则第一名会拿到 `1/k` 之外的额外偏移，公式就不对了。
- 用 `defaultdict(float)` 累加，最后 `sorted(items, key=lambda kv: kv[1], reverse=True)` 取 key。
- 边界：`rankings=[]` → `[]`；某一路返回空列表 → 只有另一路有效，正常返回该路顺序；两路都空 → `[]`。这几条要有测试。

**「去重」到底去什么（易错点）：** 去重的是**跨路重复**（同一 chunk 在向量路和 BM25 路都命中），靠 dict 的 key 天然合并。**单路内部不会有重复**（Chroma 的 `query(n_results=k)` 和 BM25 的 `sorted(...)[:top_k]` 都返回互不相同的 chunk），所以不要写额外的 `list(dict.fromkeys(...))`，那反而会把累加后的顺序打乱。

**RRF 融合原理（面试默写点）：** `RRF(d) = Σ 1/(k + rank_i(d))`，k 取 60。BM25 分和余弦相似度量纲不同，不能直接加权相加，RRF 只关心「排名」不关心「分数绝对值」，天然消除量纲差异。

## 6. 常量与 trade-off（面试讲点）

| 常量 | 值 | 为什么 |
|---|---|---|
| `VECTOR_K` | 10 | 粗筛召回要宽，避免漏 |
| `BM25_K` | 10 | 与向量对齐 |
| `CANDIDATE_K` | 8 | 融合后候选池；太小漏答案，太大 rerank 慢 |
| RRF `k` | 60 | 业界惯例 |

**精排的 trade-off（高频追问）：** Cross-Encoder 准确度比 Bi-Encoder 高约一个量级，但速度慢约 100 倍（每对 query/doc 都要过一遍模型）。所以候选池不能太大（延迟），也不能太小（漏掉正确证据）。CANDIDATE_K=8 是「先粗筛后精排」的平衡点。

## 7. `nodes/retrieve.py` 改动

```python
from functools import lru_cache

@lru_cache(maxsize=1)                      # ← 新增：进程内只构建一次
def build_retriever():
    # 1. 建 Chroma collection（现有代码不动）
    # 2. chunks = collection.get()["documents"]  ← 新增：拉全量语料建 BM25
    # 3. bm25 = build_bm25_retriever(chunks)     ← 新增
    # 4. vector_retriever = 原纯向量闭包         ← 现有，抽出复用
    # 5. reranker = CrossEncoderReranker()       ← 新增（try 失败降级 NoOpReranker）
    # 6. return build_hybrid_retriever(vector_retriever, bm25, reranker)

def retrieve_node(state, top_k=3, retriever=None):   # ← 完全不动
    ...
```

**关键卖点：** `retrieve_node` 对外签名、`state`、`contexts: list[str]` 全都不变。混合检索是 `build_retriever` 内部的升级，`generate`/`search`/`route` 无感知——这正是阶段 1「统一 contexts 结构」设计的回报。**同时因为 `build_retriever()` 有了缓存，CLI 多轮问答不会每问一次就重载 Embedding + CrossEncoder 模型。**

**缓存的三条注意：**
- `lru_cache` 要求**无参数**才最省事；现有 `build_retriever()` 正好零参，直接加装饰器即可。
- 测试中若要验证「真实构建」逻辑，需要 `build_retriever.cache_clear()` 清理；现有 18 个测试全部注入 fake，**没有一个调用真实 `build_retriever()`**，因此不会互相污染。
- 语料更新（重跑 ingest）后缓存里仍是旧 BM25 索引 → CLI 是短进程，无影响；注释里写明「长驻进程需调 `build_retriever.cache_clear()`」。

**易错点：** `collection.get()` 默认返回上限在 chromadb 新版是 10 条（实测本环境 **chromadb 1.5.9**，15 个 chunk 未带 limit 时**一次性返回了全部 15 条**，未触发截断）。为稳妥起见显式传 `limit=` 或分页；代码注释里说明「大语料需分页拉取或改落盘索引」。

**`collection.get()` 与 `query()` 的区别（写代码时别混）：** `get()` 是**全量拉取**（按 id/metadata 过滤，不做向量相似度），用来建 BM25 索引；`query()` 才是向量相似度检索。两者返回结构都有一层「批次外壳」，`get()` 取 `res["documents"]` 直接是一维 list，`query()` 要取 `res["documents"][0]`。

## 8. 测试策略

| 文件 | 测什么 | 怎么隔离 |
|---|---|---|
| `test_rrf.py` | RRF 融合纯函数，手写小 case 断言顺序 | 纯函数，无依赖 |
| `test_bm25.py` | tokenize 分词正确；精确词命中 | jieba + 小语料，无网络 |
| `test_rerank.py` | 两个 reranker 都注入 fake（假 model / 假 llm） | 不 load 真模型 |
| `test_hybrid.py` | 三组件全 fake，断言编排顺序和 top_k 截断 | 注入 fake |
| `test_retrieve.py` | **不动**，仍注入 fake retriever | 现有 |

**隔离铁律：** 真实 CrossEncoder / DeepSeek LLM / Chroma 只在 `build_retriever()` 里加载；所有测试通过注入口传 fake，**测试永不碰 HF 下载、网络 API、磁盘向量库**。

**各测试文件的具体断言（写死，避免漏测）：**

`test_rrf.py` — 纯函数，无依赖：
- 两路完全不重叠 → 按「各拿第一个名次」的顺序交错，断言具体顺序
- 同一 chunk 在 A 路第 1、B 路第 3 → 断言其总分 = `1/(60+1) + 1/(60+3)`，**且列表里只出现一次**（验证累加不重复）
- `rankings=[]` → `[]`；`[[], []]` → `[]`；只有一路 → 返回该路原顺序
- 自定义 `k`（如 `k=1`）时分数可手算断言

`test_bm25.py` — 真实 jieba（无网络、无模型）：
- `tokenize("混合检索")` 返回 list 且含非空 token
- 小语料（3~5 个手写 chunk）里查一个**只出现在其中一个 chunk 的专有词**，断言它排第一（这正是 BM25 相对向量检索的卖点，面试可演示）
- `top_k=1` 时返回长度 1；`chunks=[]` 时不崩（返回 `[]`）

`test_rerank.py` — 注入 fake：
- fake model 的 `predict(pairs)` 返回 numpy 数组（用 `numpy.array([...])` 造，**故意用 ndarray 而不是 list**，才能真的测到 `float()` 转换那条路径），断言按分数降序、截断到 `top_k`
- 断言返回的所有字符串都**来自入参 `docs`**（没被改写）
- `docs=[]` → `[]`
- `LLMReranker` 注入 fake llm：`invoke` 返回 `{"score": 0.9}` 之类，断言排序正确
- `LLMReranker` 注入会抛异常的 fake llm：断言不崩溃、该 doc 记 0 分
- `NoOpReranker`：断言原顺序 + 截断

`test_hybrid.py` — 三组件全 fake：
- 记录 fake 各组件被调用的参数，断言 `vector_retriever`/`bm25` 收到的是 `VECTOR_K`/`BM25_K`，`reranker` 收到的是**截断后的候选池**和透传的 `top_k`
- 断言最终返回值就是 `reranker` 的返回值（hybrid 不加工结果）

## 9. 依赖新增

`pyproject.toml` 的 `dependencies` 加两行：

```
"rank-bm25>=0.2.2",
"jieba>=0.42",
```

`sentence-transformers` 已存在（实测 **6.0.1**），`CrossEncoder` 直接可用，无需新增。

**已完成：** 两个包已装进 `.venv`（jieba 0.42.1 / rank-bm25 0.2.2），`pyproject.toml` 的声明待随代码一起提交。

## 9.5 模型预热（动手前必须先做）

`bge-reranker-v2-m3` **不在本地 HF 缓存**（缓存里目前只有 `models--BAAI--bge-small-zh-v1.5`），必须先下载。

**两个坑（实测踩过）：**

1. **HF 直连超时** → 用镜像 `HF_ENDPOINT=https://hf-mirror.com`。
2. **必须同时设 `HF_HUB_DISABLE_XET=1`**。huggingface_hub 1.x 默认走 Xet 存储后端，文件实际从 `cas-server.xethub.hf.co` 拉，**hf-mirror 不镜像这个域** → 报 `CAS Client Error: HTTP status client error (401 Unauthorized)`。关掉 Xet 后走传统 HTTP 通道，镜像才能覆盖。

两个环境变量都**必须在 `import huggingface_hub` 之前**设置。已封装在 `scripts/download_reranker.py`（含 `ignore_patterns` 跳过 tf/onnx/h5 权重，省流量）：

```bash
.venv/Scripts/python.exe scripts/download_reranker.py
```

下载完成后，`CrossEncoderReranker` 一律 `local_files_only=True` 离线加载。

## 10. 开发约定

- **学习式开发**：实现逻辑由用户自己动手写，助手提供接口签名、决策点、易错点，卡住时给最小片段。
- **HF 被墙**：见 §9.5，模型先下好，加载一律 `local_files_only=True`。
- **密钥/模型**：模型缓存不提交 git；`.gitignore` 已覆盖 `.env`。
- **commit 风格**：延续阶段 1 的原子 commit，一个组件一个 commit。

## 11. 验收标准

- [ ] `pytest -v` 全绿（新增 4 个测试文件 + 原 18 个不回归）
- [ ] `build_retriever()` 能真实加载 CrossEncoder（不是走 `NoOpReranker` 降级），日志无警告
- [ ] `python -m adaptive_search_rag.main`（或 CLI）真实跑通混合检索，能回答知识库问题
- [ ] 能演示「BM25 精确词命中」vs「纯向量检索」的对比（面试加分项）
- [ ] 面试能默写 RRF 公式（含 rank 从 1 起、同 doc 跨路累加）、讲清「为什么混合 + 为什么精排 + CANDIDATE_K 取值权衡」
- [ ] 能现场讲「交叉编码器 vs LLM-as-reranker」的 trade-off，且代码里两者可切换

## 12. 后续子系统（不展开，记录依赖关系）

1. 统一加载层（PyMuPDF，PDF/Markdown/文本 + 页码溯源）——nice-to-have
2. CRAG 自适应路由（先检索→评估相关性→低质量才触发搜索）——项目名核心叙事，依赖本系统的检索质量
3. RAGAS 评测（召回/忠实度/相关性）——「没量化=没做」的验收尺子，依赖前两者
