# AdaptiveSearch-RAG 设计文档

- 日期：2026-09-06
- 状态：待评审
- 定位：独立新项目（Agentic RAG），与 InvestAgent 互补

## 1. 项目概述

一个「知识库问答 Agent」，核心是把 RAG 从「无脑检索」升级为「智能决策」：

- 查询路由：判断用户问题是「知识库内知识」还是「需要联网搜索」
- 检索增强：ChromaDB 向量检索
- 联网兜底：博查（Bocha）真实联网搜索
- 生成回答：带会话记忆、断点续跑、完整 pytest 测试

### 与 InvestAgent 的区分（简历叙事）

| 项目 | 主线 | 面试关键词 |
|---|---|---|
| InvestAgent | 多智能体任务委派 + 工具调用做投研分析 | 多 Agent 编排、Supervisor、checkpoint、store |
| AdaptiveSearch-RAG | 查询路由 + 检索增强知识库 | Agentic RAG、条件路由、混合检索、重排、RAGAS |

一份讲「多 Agent 编排」，一份讲「检索增强知识库」，不是重复造 LangGraph 玩具。

## 2. 技术栈

| 层 | 选型 | 复用来源 |
|---|---|---|
| LLM | DeepSeek V4（`deepseek-v4-flash`） | investagent 已验证 |
| Embedding | `BAAI/bge-small-zh-v1.5`（本地） | farmer-help 已验证 |
| 编排 | LangGraph `StateGraph` + `conditional_edges` | investagent |
| 向量库 | ChromaDB（余弦相似度） | farmer-help |
| checkpoint | `SqliteSaver`（会话记忆 + 断点续跑） | investagent 阶段3 |
| 联网搜索 | 博查 Bocha AI Search API（key 已获取） | 新增 |
| 文档加载 | 阶段1 Markdown；阶段2 统一加载层（PyMuPDF） | 新增 |
| 测试 | pytest | investagent |

唯一的新东西：博查搜索工具 + 查询路由逻辑。其余全部复用已有经验。

## 3. 阶段划分

### 阶段 1：核心链路（先路由模式）

目标：快速把 `路由 → 检索/搜索 → 汇总` 整条状态机跑通，写单元测试，不纠结效果最优。

- 文档：Markdown（快速验证）
- 路由：先路由模式（route 先判断来源，再决定走检索还是搜索）
- 搜索：直接接博查真实 API，测试时 mock 掉搜索工具
- 交付：CLI 脚本 + pytest 全绿

### 阶段 2：升级为自适应检索（CRAG）

- 文档：统一加载层（PyMuPDF 支持 PDF/Markdown/纯文本 + 页码溯源）
- 路由：升级为「先检索 → 评估质量 → 条件触发搜索」CRAG 自适应 RAG
- 检索：混合检索（BM25 + 向量）+ rerank 重排
- 评测：RAGAS 评测召回/答案质量
- README 写清楚「先路由 → 自适应检索」的迭代演进

> 本设计文档聚焦阶段 1 的实现细节；阶段 2 只记方向，届时单独展开设计。

## 4. State 与图结构（阶段 1）

### State

用普通 `TypedDict`（不用 MessagesState——本项目是「问题→上下文→答案」的单向数据流，不是多轮对话）。

```python
class RAGState(TypedDict):
    question: str                # 用户问题
    route: str                   # "retrieve" | "search"
    contexts: list[str]          # 检索或搜索到的文本片段（统一结构）
    answer: str                  # 最终回答
```

### 图结构

```
START → route ──(conditional_edges)──┬→ retrieve → generate → END
                                     └→ search   → generate → END
```

4 个节点：`route` / `retrieve` / `search` / `generate`。

**关键设计点**：`retrieve` 和 `search` 两个节点输出**统一的 `contexts` 结构**（`list[str]`），让 `generate` 节点完全不关心上下文来自文档还是网络——这是让图保持简单、可测试的核心。

## 5. 节点设计

### 5.1 route（路由节点）

- 输入：`question`
- 动作：LLM 判断问题是「知识库内知识」还是「需要联网搜索」
- 输出：`route` 字段（`"retrieve" | "search"`）
- 实现：`with_structured_output` 输出枚举（复用 investagent 阶段4b 经验：`method="function_calling"` + `extra_body={"thinking":{"type":"disabled"}}`）

### 5.2 retrieve（检索节点）

- 输入：`question`
- 动作：ChromaDB 相似度检索 top-k，取 `content` 拼成 `contexts`
- 输出：`contexts`

### 5.3 search（搜索节点）

- 输入：`question`
- 动作：调博查 `web_search(query)`，把返回的 `{title, content, url}` 规整成统一格式
- 输出：`contexts`（与 retrieve 同结构）

### 5.4 generate（生成节点）

- 输入：`question` + `contexts`
- 动作：把 `contexts` 拼进 prompt，LLM 生成 `answer`
- 输出：`answer`

## 6. 数据流

```
用户问 → route 判断来源 → 取 context（文档 / 网络）→ 拼 prompt → 生成回答
```

## 7. 错误处理与边界

| 边界 | 处理 |
|---|---|
| 知识库为空 / 未入库 | retrieve 返回空 → generate 明确说「知识库暂无相关内容」，不硬编 |
| 检索无结果 | `contexts` 为空 → generate 引导用户（阶段2 触发搜索兜底） |
| 博查 API 失败 | search 节点 try/except → 返回「联网搜索失败」降级，不让图崩 |
| 路由 LLM 返回非法值 | 在 `path_fn` 里对非法值做 fallback（归入 retrieve），避免 conditional_edges 因返回值未命中 path_map 而报错 |

## 8. 测试策略

| 层 | 怎么测 |
|---|---|
| 节点级 | route/retrieve/generate 用 fake LLM + fake retriever 单测（不碰真实 API） |
| 搜索工具 | mock 掉博查（注入假搜索函数），复用 investagent 阶段4a「测试替身」思路 |
| 图级 | 用 InMemorySaver + 假节点，断言「问文档内问题 → 走 retrieve 分支」 |
| 边界 | 空库、检索空、搜索失败三条路径各有测试 |

## 9. 项目结构（src 布局）

```
AdaptiveSearch-RAG/
├── src/adaptive_search_rag/
│   ├── state.py            # RAGState
│   ├── graph.py            # build_graph() 组装图
│   ├── nodes/
│   │   ├── route.py        # 路由节点
│   │   ├── retrieve.py     # 检索节点
│   │   ├── search.py       # 搜索节点
│   │   └── generate.py     # 生成节点
│   ├── tools/web_search.py # 博查封装
│   ├── ingest.py           # 文档加载 + 分块 + 入库
│   └── main.py             # CLI 入口
├── tests/
├── .env / .env.example
└── pyproject.toml
```

> 项目目录名 `AdaptiveSearch-RAG`（连字符），Python 包名 `adaptive_search_rag`（下划线，Python 不允许连字符）。

## 10. 开发约定

- **学习式开发**：代码由用户自己动手写，助手提供思路、设计、关键 API 用法和「为什么」；卡住时给最小片段。
- **环境**：PyCharm（venv 在 PyCharm 里创建，Run Configuration 运行）。
- **密钥安全**：`.env` 加入 `.gitignore`，提供 `.env.example` 占位，首次 commit 前验证 `git check-ignore .env`。

## 11. 阶段 2 方向（不展开）

1. 统一加载层：PyMuPDF 支持 PDF/Markdown/文本，保留页码元数据用于答案溯源
2. 路由升级：先检索 → 评估相关性 → 低质量才触发搜索（CRAG / Self-RAG）
3. 混合检索：BM25（稀疏）+ 向量（稠密），再 rerank 重排
4. 评测：RAGAS 评测召回率、答案忠实度、相关性
