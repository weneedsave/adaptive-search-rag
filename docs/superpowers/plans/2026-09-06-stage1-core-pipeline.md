# AdaptiveSearch-RAG 阶段1 实现计划（先路由模式）

> **执行方式：学习式开发（指导式）**。本计划是「指导式」而非「抄写式」——
> 接口签名、测试要点、关键 API 提示给精确，**实现代码由用户自己写**。
> 配置类脚手架（pyproject / .env / .gitignore）不属于学习点，直接给出内容。
> 每个任务用 checkbox 跟踪，完成一项勾一项。

**Goal:** 跑通「路由 → 检索/搜索 → 汇总」核心链路（图内 4 节点 + 独立 ingest 入库脚本），配完整 pytest，产出可放 GitHub 的 Agentic RAG 项目。

**Architecture:** LangGraph `StateGraph` 组装 4 节点（route/retrieve/search/generate），`conditional_edges` 做条件路由；ChromaDB 向量检索 + 博查真实联网搜索，输出统一 `contexts` 结构给生成节点；`SqliteSaver` 做会话记忆。

**Tech Stack:** LangGraph + langgraph-checkpoint-sqlite + langchain-openai（DeepSeek V4）+ ChromaDB + sentence-transformers（bge-small-zh）+ httpx（博查）+ pytest。

**Spec:** `docs/superpowers/specs/2026-09-06-adaptive-search-rag-design.md`

## Global Constraints（全局约束，每个任务都隐含遵守）

- Python 3.13，项目目录 `E:\Ai\agent\AdaptiveSearch-RAG\`，包名 `adaptive_search_rag`（下划线）
- LLM：`deepseek-v4-flash`，必须关 thinking：`extra_body={"thinking":{"type":"disabled"}}`
- Embedding：`BAAI/bge-small-zh-v1.5`，本地加载 `local_files_only=True`（HF 被墙）
- ChromaDB 集合名 `adaptive_knowledge`，余弦相似度
- 环境：PyCharm（venv 在 PyCharm 里创建，Run Configuration 运行）
- 密钥安全：`.env` 不提交，提供 `.env.example`；首次 commit 前验证 `git check-ignore .env`
- 学习式开发：用户自己写实现代码，助手只给思路/接口/API 提示
- 每次提交前跑对应测试，全绿再 commit

---

### Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/adaptive_search_rag/__init__.py`
- Create: `tests/__init__.py`
- 初始化：git 仓库 + PyCharm venv（`.venv/`）

**Interfaces:**
- Produces: 可安装的包 + 环境，后续所有任务的基础

- [ ] **Step 1: 初始化 git 仓库**

在 `E:\Ai\agent\AdaptiveSearch-RAG\` 下 `git init`（本地即可，暂不推远程）。

- [ ] **Step 2: 写 `pyproject.toml`**（直接照抄，这是配置非学习点）

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "adaptive-search-rag"
version = "0.1.0"
description = "Agentic RAG with query routing and web search"
requires-python = ">=3.13"
dependencies = [
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "langchain-openai",
    "langchain-core",
    "chromadb",
    "sentence-transformers",
    "httpx",
    "python-dotenv",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: 写 `.gitignore`**（照抄）

```
.venv/
__pycache__/
*.pyc
.env
*.sqlite
chroma_db/
.pytest_cache/
```

- [ ] **Step 4: 写 `.env.example`**（照抄，占位符）

```
# DeepSeek
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

# Bocha 搜索
BOCHA_API_KEY=sk-your-bocha-key
```

- [ ] **Step 5: 在 PyCharm 里建 venv + 安装**

PyCharm 打开项目 → 用 Python 3.13 建 `.venv/` → 终端里 `pip install -e .`。

- [ ] **Step 6: 填 `.env` 真实 key**

复制 `.env.example` 为 `.env`，填入真实 DeepSeek key + Bocha key。跑 `git check-ignore .env` 确认被忽略。

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold project"
```

---

### Task 2: 定义 RAGState

**Files:**
- Create: `src/adaptive_search_rag/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Produces: `RAGState`（TypedDict，字段见下）

```python
class RAGState(TypedDict):
    question: str      # 用户问题
    route: str         # "retrieve" | "search"
    contexts: list[str] # 统一上下文（文档或网络来源）
    answer: str        # 最终回答
```

- [ ] **Step 1: 写测试**（`tests/test_state.py`）

写一个测试：构造一个 `RAGState` 字典实例，断言能读回四个字段。思考：TypedDict 本质是 dict，运行时怎么断言「字段类型」？答案是——测试里主要验证「字段名存在且能赋值读取」，以及用 `typing.get_type_hints` 断言字段类型声明正确。

- [ ] **Step 2: 跑测试确认失败**

`pytest tests/test_state.py -v`，预期失败（`state.py` 不存在）。

- [ ] **Step 3: 实现 `state.py`**（你写）

提示：从 `typing` 导入 `TypedDict`，按上面签名定义。想一想为什么用普通 TypedDict 而不是 `MessagesState`（答案见 spec §4：单向数据流，非多轮对话）。

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** `feat: define RAGState`

---

### Task 3: 文档加载 + 分块 + 入库（ingest）

**Files:**
- Create: `src/adaptive_search_rag/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `RAGState`（间接，不依赖）
- Produces:
```python
def load_markdown(path: str) -> str
def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]
def ingest_directory(dir_path: str, collection_name: str = "adaptive_knowledge") -> int  # 返回入库 chunk 数
```

- [ ] **Step 1: 写测试**（`tests/test_ingest.py`）

测三件事：
1. `split_text` 把一段长文本切成多块，且块之间有 overlap（断言第一块结尾 == 第二块开头的一部分）
2. `load_markdown` 读一个临时 `.md` 文件返回非空字符串
3. `ingest_directory` 对一个含 1 个 md 文件的临时目录返回 chunk 数 > 0（用临时的 collection_name，避免污染主库）

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 `ingest.py`**（你写）

关键提示：
- 分块：最简单的按字符切（`chunk_size` 步长 `chunk_size - chunk_overlap`），阶段1 不必上复杂 splitter
- 向量化：用 `sentence_transformers` 加载 `BAAI/bge-small-zh-v1.5`，`local_files_only=True`
- 入库：ChromaDB `chromadb.Client()`（非持久化）或 `PersistentClient(path="chroma_db/")`——思考差异，阶段1 建议用 PersistentClient 让数据落盘
- 每条 chunk 用 `collection.add(documents=[...], embeddings=[...], ids=[...])`，id 要唯一（用文件名+序号）

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: 准备示例语料**

新建 `data/` 目录，放 2-3 个你自己写的 Markdown 文档（如 `data/rag-intro.md` 讲 RAG 原理、`data/langgraph.md` 讲 LangGraph 概念）。这是后面手动验证的语料。

- [ ] **Step 6: Commit** `feat: add markdown ingest`

---

### Task 4: 检索节点（retrieve）

**Files:**
- Create: `src/adaptive_search_rag/nodes/__init__.py`
- Create: `src/adaptive_search_rag/nodes/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `ingest.py` 的入库结果（ChromaDB collection）
- Produces:
```python
def retrieve_node(state: RAGState, top_k: int = 3) -> dict:
    """返回 {"contexts": [...]}"""
```

- [ ] **Step 1: 写测试**（`tests/test_retrieve.py`）

测试：先往临时 collection 塞几条已知文本，构造 `RAGState(question=...)`，调用 `retrieve_node`，断言返回的 `contexts` 非空、且是 `list[str]`。再测「空库」：对空 collection 检索，断言 `contexts == []`（这是 spec §7 的边界）。

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 `retrieve.py`**（你写）

关键提示：
- 用同一个 embedding 模型把 `question` 向量化，再 `collection.query(query_embeddings=[...], n_results=top_k)`
- `query` 返回的 `documents` 是 `list[list[str]]`（外层是查询数，内层是结果数），注意取第一层
- 返回统一的 `{"contexts": [...]}`，空库或低相似度时返回空列表

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** `feat: add retrieve node`

---

### Task 5: 博查搜索工具（web_search）

**Files:**
- Create: `src/adaptive_search_rag/tools/__init__.py`
- Create: `src/adaptive_search_rag/tools/web_search.py`
- Test: `tests/test_web_search.py`

**Interfaces:**
- Produces:
```python
def web_search(query: str, top_k: int = 3) -> list[str]:
    """调博查 API，返回规整后的文本片段列表（每段含标题+摘要+链接）"""
```

- [ ] **Step 1: 写测试**（`tests/test_web_search.py`）

**用 mock，不碰真实 API**（这是关键学习点）。用 `unittest.mock` 或 monkeypatch 把 httpx 请求 mock 掉，返回假的博查响应 JSON。断言：
1. `web_search("测试")` 返回 `list[str]`，长度 == top_k
2. 每条片段里包含标题和链接信息

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 `web_search.py`**（你写）

关键提示：
- 博查 API（以官方文档为准，大致格式）：
  - POST `https://api.bochaai.com/v1/web-search`
  - Header：`Authorization: Bearer <BOCHA_API_KEY>`、`Content-Type: application/json`
  - Body：`{"query": "...", "count": top_k, "summary": true, "freshness": "noLimit"}`
  - 返回：`data.webPages.value` 数组，每项含 `name`/`url`/`summary`/`snippet`
- 用 `httpx` 发请求，`.env` 里读 key（`python-dotenv` 的 `load_dotenv`）
- 规整：把每条拼成 `f"[{name}]({url})\n{summary}"` 这类统一格式塞进列表

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** `feat: add bocha web search tool`

---

### Task 6: 搜索节点（search）

**Files:**
- Create: `src/adaptive_search_rag/nodes/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `tools/web_search.py` 的 `web_search(query, top_k)`
- Produces:
```python
def search_node(state: RAGState, top_k: int = 3) -> dict:
    """返回 {"contexts": [...]}，与 retrieve 同结构"""
```

- [ ] **Step 1: 写测试**（`tests/test_search.py`）

**mock 掉 `web_search`**（注入假函数，复用 investagent 阶段4a「测试替身」思路）。测：
1. 正常：mock 返回固定列表 → `search_node` 返回的 `contexts` == 该列表
2. 失败降级：mock 抛异常 → `search_node` 不崩，返回 `{"contexts": []}`（或含「搜索失败」标记），这是 spec §7 的边界

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 `search.py`**（你写）

关键提示：`try/except` 包住 `web_search` 调用，失败时降级返回空 contexts，不让图崩。

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** `feat: add search node`

---

### Task 7: 路由节点（route）

**Files:**
- Create: `src/adaptive_search_rag/nodes/route.py`
- Test: `tests/test_route.py`

**Interfaces:**
- Consumes: DeepSeek LLM（`langchain-openai`）
- Produces:
```python
def route_node(state: RAGState) -> dict:
    """判断问题属于知识库还是联网，返回 {"route": "retrieve" | "search"}"""
```

- [ ] **Step 1: 写测试**（`tests/test_route.py`）

**mock 掉 LLM**（这是关键：测试不碰真实 API）。用 fake LLM（返回固定结构化结果）。测：
1. LLM 返回「文档内」→ `route` == `"retrieve"`
2. LLM 返回「联网」→ `route` == `"search"`

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 `route.py`**（你写）

关键提示（复用 investagent 阶段4b 经验，面试必考）：
- 定义枚举 `RouteDecision`，用 `Literal["retrieve", "search"]` 或 Pydantic 模型
- 用 `llm.with_structured_output(RouteDecision, method="function_calling")`
- DeepSeek 三坑：老字段名 `model_name`/`openai_api_base`/`openai_api_key`；模型 `deepseek-v4-flash`；**关 thinking**：`extra_body={"thinking":{"type":"disabled"}}`（否则 `with_structured_output` 报错）
- 思考：LLM 判断「文档内 vs 联网」的 prompt 该怎么写才准？——告诉它「如果问题能在你（作为知识库）已知的知识里回答，选 retrieve；如果涉及实时/最新/特定事实查询，选 search」

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** `feat: add route node`

---

### Task 8: 生成节点（generate）

**Files:**
- Create: `src/adaptive_search_rag/nodes/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `RAGState` 的 `question` + `contexts`、DeepSeek LLM
- Produces:
```python
def generate_node(state: RAGState) -> dict:
    """返回 {"answer": ...}"""
```

- [ ] **Step 1: 写测试**（`tests/test_generate.py`）

**mock 掉 LLM**。测：
1. 有 contexts：prompt 里应包含 question 和 contexts 内容（mock 的 LLM 能捕获输入 prompt 供断言）
2. 无 contexts（`contexts == []`）：generate 返回「知识库暂无相关内容」类提示（spec §7 边界）

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 `generate.py`**（你写）

关键提示：
- 拼 prompt：`SystemMessage`（设定「基于给定资料回答，资料没有就说不知道，不要编造」）+ `HumanMessage`（question + 拼接的 contexts）
- 无 contexts 时，走「明确告知无相关内容」的分支，不硬编

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** `feat: add generate node`

---

### Task 9: 组装图 + 条件路由（graph）

**Files:**
- Create: `src/adaptive_search_rag/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: 四个节点函数 + `RAGState`
- Produces:
```python
def build_graph(checkpointer=None) -> CompiledGraph:
    """组装图并 compile，返回可 invoke 的图"""
```

- [ ] **Step 1: 写测试**（`tests/test_graph.py`）

**图级集成测试**（用 fake 节点，不碰真实 LLM/向量库）。测：
1. 构造一个 `build_graph`，传入 fake 的 route/retrieve/generate（思考：怎么把节点注入？——给 `build_graph` 加可选的 `nodes` 参数，或 monkeypatch 节点模块），断言「route 返回 retrieve → 图走 retrieve 分支 → 最终 answer 正确」
2. 同理断言「route 返回 search → 走 search 分支」

- [ ] **Step 2: 跑测试确认失败**

- [ ] **Step 3: 实现 `graph.py`**（你写）

关键提示（LangGraph 核心，面试必考）：
```python
g = StateGraph(RAGState)
g.add_node("route", route_node)
g.add_node("retrieve", retrieve_node)
g.add_node("search", search_node)
g.add_node("generate", generate_node)
g.add_edge(START, "route")
g.add_conditional_edges("route", path_fn, {"retrieve": "retrieve", "search": "search"})
g.add_edge("retrieve", "generate")
g.add_edge("search", "generate")
g.add_edge("generate", END)
return g.compile(checkpointer=checkpointer)
```
- `path_fn` 就是 `lambda state: state["route"]`
- **非法值兜底**：`path_fn` 里对不是 `retrieve`/`search` 的值 fallback 到 `retrieve`（spec §7，否则 conditional_edges 报错）
- 思考：为什么 route 后两个分支都要 `add_edge` 到 generate（汇合点）

- [ ] **Step 4: 跑测试确认通过**

- [ ] **Step 5: Commit** `feat: assemble graph with conditional routing`

---

### Task 10: CLI 入口 + 会话记忆（checkpointer）

**Files:**
- Create: `src/adaptive_search_rag/main.py`
- Modify: `src/adaptive_search_rag/graph.py`（接入 SqliteSaver）
- Test: `tests/test_main.py`（可选，CLI 侧重手动验证）

**Interfaces:**
- Produces:
```python
def ask(question: str, thread_id: str = "default") -> str:
    """带 checkpointer 的问答，返回 answer"""

def main() -> None:
    """CLI：读 stdin/argv 的问题，打印答案"""
```

- [ ] **Step 1: 接入 SqliteSaver**（你写）

关键提示（复用 investagent 阶段3 经验，面试必考）：
- `from langgraph.checkpoint.sqlite import SqliteSaver`
- `SqliteSaver.from_conn_string("checkpoints.sqlite")` 是**上下文管理器**（with 块）
- `build_graph(checkpointer=saver)` 把 saver 传给 compile
- 挂上 checkpointer 后，**每次 invoke 必须传 `config={"configurable": {"thread_id": ...}}`**，否则报 `ValueError`

- [ ] **Step 2: 实现 `ask` + `main`**（你写）

关键提示：
- `ask`：先 `load_dotenv()`，组装带 checkpointer 的图，`g.invoke({"question": q}, config)`，返回 `state["answer"]`
- `main`：CLI 循环，读输入 → ask → 打印（Windows git bash 下注意 UTF-8 输出，参考 farmer-help 经验）
- 用不同 `thread_id` 演示「同一会话有记忆、不同会话隔离」

- [ ] **Step 3: 手动端到端验证**

跑通完整链路：先 `python -m adaptive_search_rag.main` 或脚本，问一个「文档内问题」（如你语料里的 RAG 原理）应走 retrieve；问一个「联网问题」（如「今天某股票股价」）应走 search。截图留档。

- [ ] **Step 4: Commit** `feat: add CLI and checkpoint memory`

---

### Task 11: 边界处理完善 + 全量测试

**Files:**
- Modify: 各节点（补边界分支）
- Test: 补充边界测试用例

**Interfaces:**
- 无新增，完善已有

- [ ] **Step 1: 补边界测试**

对照 spec §7 逐条确认有测试覆盖：空库、检索无结果、搜索失败、路由非法值兜底。缺哪个补哪个。

- [ ] **Step 2: 跑全量测试**

`pytest -v`，确认全绿。

- [ ] **Step 3: 写 README**（你写，这是简历门面）

README 结构建议：项目一句话简介 → 架构图（文字版或 drawio）→ 快速开始（装依赖、填 .env、入库、问答）→ 项目结构 → 测试方法 → **阶段2 演进方向（先路由 → CRAG）**。

- [ ] **Step 4: 最终 Commit + 推 GitHub**

```bash
git add -A
git commit -m "docs: add README and finalize stage1"
```
（是否推远程你决定，推之前确认 `.env` 不在追踪里）

---

## 自审记录

- **Spec 覆盖**：spec §4（State/图）→ Task 2/9；§5（4节点）→ Task 4/6/7/8；§7（边界）→ Task 4/6/8/9/11；§8（测试策略）→ 各任务 Step1；§9（目录）→ Task 1 脚手架 + 各任务；§10（约定）→ Task 1/11。
- **类型一致性**：`RAGState` 四字段（question/route/contexts/answer）在 Task 2 定义，后续任务一致引用；`contexts` 统一为 `list[str]`，retrieve/search 输出同构；`build_graph(checkpointer=None)` 签名 Task 9 定义、Task 10 复用。
- **无占位符**：配置类直接给全文；核心逻辑给接口 + 关键 API 提示 + 思考点（符合学习式开发）。

## 执行交接

计划完成。由于是**学习式开发**，不用 subagent/inline 自动执行——**明天开始，我按任务顺序逐条指导你**：每个任务先讲「为什么 + 怎么做 + 关键 API」，你写实现和测试，卡住我给你最小片段，跑通再进下一个。

今天到此为止。明天开工时说一声「开始 Task 1」即可。
