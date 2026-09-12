# AdaptiveSearch-RAG 自适应检索问答系统

基于 LangGraph 的 Agentic RAG：先检索本地知识库，再由 LLM 三档评估资料是否足够，不足则自动改写查询重试或联网搜索兜底；支持 SQLite 会话持久化与完整单元测试。

## ✨ 项目简介

1. **自适应检索（CRAG）**：先检索，LLM 三档评估资料是否足够（`correct` / `ambiguous` / `incorrect`）。足够直接作答；部分相关补充联网；完全无关则改写查询重试本地，仍不达标才联网。
2. **混合检索**：向量（稠密）+ BM25（稀疏）双路粗筛 → RRF 融合 → CrossEncoder 精排，召回本地知识库相关片段。
3. **联网兜底**：博查（Bocha）真实联网搜索，规整为统一上下文结构。
4. **会话持久化**：LangGraph SqliteSaver 保存对话状态，`thread_id` 区分会话，支持会话隔离。
5. **容错兜底**：评估结果非法时兜底为 `incorrect`；改写失败回退原问题且计数照增（保证重试环必定终止）；联网失败保留本地结果；空上下文不调用 LLM，减少幻觉。
6. **全量单元测试**：依赖注入 + mock，绝大多数测试无需真实 LLM / 向量库 / 网络。

## 🛠️ 技术栈

| 层 | 选型 |
| --- | --- |
| 编排 | LangGraph（StateGraph + conditional_edges） |
| 会话记忆 | langgraph-checkpoint-sqlite（SqliteSaver） |
| LLM | DeepSeek V4（`deepseek-v4-flash`，langchain-openai） |
| Embedding | BAAI/bge-small-zh-v1.5（sentence-transformers，本地离线） |
| 向量库 | ChromaDB（PersistentClient，余弦相似度） |
| 稀疏检索 | rank-bm25 + jieba（中文分词） |
| 精排 | BAAI/bge-reranker-v2-m3（sentence-transformers CrossEncoder，本地离线） |
| 联网搜索 | 博查 Bocha AI Search API（httpx） |
| 后端语言 | Python 3.13 |

## 📁 目录结构

```
AdaptiveSearch-RAG/
├── src/adaptive_search_rag/
│   ├── state.py          # RAGState 六字段（question/retrieval_query/contexts/grade/retry_count/answer）
│   ├── llm.py            # DeepSeek LLM 构造（关 thinking）
│   ├── ingest.py         # 文档加载 + 分块 + 向量化入库
│   ├── graph.py          # LangGraph 图组装 + CRAG 环拓扑 + should_act 条件边
│   ├── main.py           # CLI 交互入口、ask 问答函数
│   ├── nodes/
│   │   ├── evaluate.py   # 评估节点：LLM 三档裁定资料是否足够
│   │   ├── rewrite.py    # 改写节点：疑问句 → 关键词（治中文 BM25 的句式干扰）
│   │   ├── retrieve.py   # 本地知识库检索节点（组装混合检索 + lru_cache）
│   │   ├── search.py     # 联网搜索节点（结果追加而非覆盖，失败保留本地）
│   │   └── generate.py   # 答案生成节点（空上下文兜底）
│   ├── retrieval/        # 混合检索子系统
│   │   ├── bm25.py       # 中文 BM25（jieba 分词 + rank_bm25）
│   │   ├── rerank.py     # 三个 reranker：CrossEncoder / LLM / NoOp 降级
│   │   └── hybrid.py     # 常量 + RRF 融合 + 三组件编排
│   └── tools/
│       └── web_search.py # 博查 API 封装
├── data/                 # 知识库语料（Markdown）
├── tests/                # 单元测试（11 个测试文件，52 个用例）
├── scripts/              # 手动验证 / 演示脚本
├── .env.example          # 环境变量模板
└── pyproject.toml
```

## 📦 安装依赖

```bash
pip install -e .
```

（`pyproject.toml` 声明了全部依赖；editable 模式，改代码无需重装。）

## ⚙️ 环境配置

复制 `.env.example` 为 `.env`，填入真实密钥：

```
# DeepSeek
DEEPSEEK_API_KEY=sk-你的deepseek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

# 博查搜索
BOCHA_API_KEY=sk-你的bocha密钥
```

> `.env` 已加入 `.gitignore`，不会提交。

## 🚀 运行项目

### 1. 入库（首次必做）

将 `data/` 下的 Markdown 语料向量化写入 ChromaDB：

```bash
python -c "from adaptive_search_rag.ingest import ingest_directory; ingest_directory('data')"
```

### 2. 下载精排模型（首次必做）

`bge-reranker-v2-m3`（约 2.3GB）需先下到本地缓存。国内直连 HuggingFace 不通，脚本已内置 hf-mirror 镜像 + 关闭 Xet 后端：

```bash
python scripts/download_reranker.py
```

> 模型加载一律 `local_files_only=True` 离线进行。若加载失败，系统会自动降级为 `NoOpReranker`（打印警告，检索链路仍可用）。

### 3. 交互式命令行问答

```bash
python -m adaptive_search_rag.main
```

输入 `exit` / `quit` 退出。

### 4. 验证与演示脚本

```bash
# 探活：确认精排真实生效、lru_cache 命中、模型加载耗时
python scripts/smoke_hybrid.py

# 对比演示：纯向量 vs BM25 vs 混合+精排 三路结果对照
python scripts/demo_hybrid.py

# 图链路验证：基础问答、同 thread 会话持久化、不同 thread 会话隔离
python scripts/verify_main.py

# CRAG 端到端路径观察：打印三档判定、实际执行路径、改写结果与耗时
# （retrieve 出现两次 = 触发了重试环。真实调用 LLM + 检索 + 联网，约 1~2 分钟）
python scripts/verify_graph.py

# 单节点探针：改写节点的疑问句 → 关键词实际输出
python scripts/verify_rewrite.py

# 单节点探针：评估节点的三档判定实际输出
python scripts/verify_evaluate.py
```

## 🧪 运行单元测试

```bash
pytest -v
```

> 绝大多数测试通过依赖注入假节点 / mock 掉网络，无需真实 LLM、向量库、网络，快速验证逻辑。
> 例外：`test_ingest_directory` 会真实加载本地 embedding 模型（约 20s）。
>
> 检索层的三个组件（向量 / BM25 / reranker）在 `build_hybrid_retriever` 处依赖注入，测试全部传假对象——**测试永不触发 HF 下载、网络 API 或磁盘向量库**。

## 📌 核心流程

先检索，拿到证据后再判断够不够——而不是在检索前猜。

```
START → retrieve → evaluate ─┬─ "correct"   ──────────────→ generate → END
                             ├─ "ambiguous" ──→ search ───→ generate
                             └─ "incorrect" → rewrite ──┐
                                          ↑              │
                                          └──────────────┘  回到 retrieve
                                            （上限 1 次，撞上限转 search）
```

`retrieve` 有两个入边（`START` 和 `rewrite`），**这条回边就是「环」**——也是 LangGraph 相对手写 `if/else` 真正体现价值的场景。

关键设计：

- **统一 contexts 结构**：retrieve 与 search 输出同构 `{"contexts": list[str]}`，generate 不关心上下文来源。
- **`question` 与 `retrieval_query` 分离**：改写只影响检索，不污染生成和联网。联网始终用用户原问题，本地检索用改写后的关键词。
- **职责放在唯一知道答案的节点上**：评估器判 `incorrect` 时**主动清空 contexts**，于是 search 可以无脑 `existing + new` 拼接，不必自己判断该不该融合。
- **环的封顶**：`retry_count` 由改写节点**无条件**自增（LLM 失败也增），保证环必定终止。
- **搜索降级**：博查调用失败 → 保留本地已有 contexts，不中断图执行。
- **防幻觉**：contexts 为空时，generate 不调用 LLM，直接返回「根据现有资料无法回答」。

## 🔄 架构演进：从「先路由」到「后置评估」

阶段 1 用 `route` 节点在**检索之前**判断来源。它有一个结构性问题：
**`search` 分支会绕过一切评估**——系统里于是有两条通往联网的路，一条经过评估、一条没有，
你无法回答「它到底会不会乱联网」。

阶段 2 改为 CRAG：**不再猜，先检索，拿到证据再判断。**
路由能力没有消失，它搬进了 `evaluate`，从「猜」变成了「看证据」。

> 引入 route：`535b53b feat: add route node`
> 移除 route：`refactor: replace pre-retrieval route with CRAG evaluation loop`

**代价（实测）**：明显该联网的问题（如实时天气）现在要空跑两轮本地检索，
端到端 **17s**，而旧架构约 **2s** 直连联网。这是有意识的取舍。
优化路径是加一个**不需要 LLM 的关键词前置门**（纯筛「今天/天气/股价」这类字面无争议的实时词），
而**不是**把 `route` 加回来——那会重新引入不可评估的后门。

## 🔍 检索层：混合检索 + 精排

`retrieve_node` 内部是「双路粗筛 → 融合 → 精排」三段式。对外接口不变（仍是 `retriever(question, top_k) -> list[str]`）。

```
question
  ├─→ 向量检索 top-10  (Chroma，语义召回)
  └─→ BM25  检索 top-10 (内存索引 + jieba，精确词召回)
          ↓ RRF 融合（k=60，量纲不同不能直接加权）
        候选池 top-8
          ↓ CrossEncoder 精排（query+doc 深度交互）
        top-3 → contexts
```

### 为什么必须混合

两路检索的失效场景**恰好互补**，实测（`scripts/demo_hybrid.py`）可见：

| 问题类型 | 向量检索 | BM25 |
| --- | --- | --- |
| 「StateGraph 是什么」（精确术语） | 一般 | **强** |
| 「什么是 RAG 的幻觉问题」（语义型） | **强** | 一般 |

纯向量检索对专有名词、缩写、编号不敏感；纯 BM25 反问句/口语化提问时会被问题句式带偏。

### 为什么用 RRF 而不是加权求和

BM25 的分数是词频得分（可到几十），向量检索是余弦相似度（0~1），**量纲不同，`0.7×向量 + 0.3×BM25` 里的权重无法可解释地确定**。

RRF 只关心「排名」不关心「分数绝对值」，天然消除量纲差异：

```
RRF(d) = Σᵢ 1 / (k + rankᵢ(d))      k = 60，rank 从 1 起
```

同一文档在多路里命中时**分数累加**——这正是 RRF 奖励「多路共识」的机制。

### 精排的代价

Cross-Encoder 把 `[query, doc]` 拼成一对送进模型做深度交互，准确度远高于双塔，但**速度慢得多**。本机 CPU 实测：

| 阶段 | 耗时 |
| --- | --- |
| 向量检索 top-10 | 9.3 ms |
| BM25 检索 top-10 | 0.2 ms |
| RRF 融合 | 9.9 ms |
| **CrossEncoder 精排 8 条** | **~5400 ms** |

精排占总耗时 **99.6%**，且**随文档长度显著增长**（12 字符 370ms → 540 字符 6467ms）。这正是它只对 `CANDIDATE_K=8` 的小候选池运行的原因——候选池取 100 会让单次问答等到 60 秒以上。

### 三个可切换的 Reranker

共用同一签名 `rerank(question, docs, top_k) -> list[str]`，hybrid 层不关心背后是哪个：

| 实现 | 说明 |
| --- | --- |
| `CrossEncoderReranker` | **默认**。bge-reranker-v2-m3，懒加载 + 离线 |
| `LLMReranker` | 逐条让 LLM 打分。单条失败记 0 分不中断整体 |
| `NoOpReranker` | **降级用**。CrossEncoder 加载失败时启用，按 RRF 原顺序截断，保证链路不断 |

### 关键常量

| 常量 | 值 | 取舍 |
| --- | --- | --- |
| `VECTOR_K` / `BM25_K` | 10 / 10 | 粗筛要宽，避免正确答案进不了候选池 |
| `CANDIDATE_K` | 8 | 太小漏答案，太大精排慢（每多一条 +700ms） |
| `RRF_K` | 60 | 业界惯例 |

## ❓ 常见问题

1. Windows 控制台中文乱码：代码内置 `sys.stdout.reconfigure(encoding="utf-8")` 自动处理。
2. 清空所有对话记忆：删除项目里的 `checkpoints.sqlite` 文件。
3. 单元测试大多不依赖 `.env`（用假节点）；仅 `test_ingest_directory` 需本地有 embedding 模型缓存。

## 📈 阶段 2 进展

- ✅ **混合检索 + 精排**：BM25（稀疏）+ 向量（稠密）→ RRF 融合 → CrossEncoder 重排。
- ✅ **CRAG 自适应检索**：先检索 → 三档评估 → 质量不足才触发联网（见上方「核心流程」）。

### 实测（真实 LLM + 真实混合检索 + 真实联网）

| 用例 | 执行路径 | grade | retry | 耗时 |
| --- | --- | --- | --- | --- |
| `什么是RAG？` | retrieve → evaluate → generate | correct | 0 | 9.0s |
| `今天北京天气怎么样？` | retrieve → evaluate → rewrite → retrieve → evaluate → search → generate | incorrect | 1 | 17.0s |

改写效果实测：`今天北京天气怎么样？` → `北京 天气`

**已知的模糊地带**：`ambiguous` 与 `incorrect` 的边界天然模糊（「提到了 X 但没展开」判哪档都说得通）。
容错设计弥补了它——判 `incorrect` 多跑一轮改写再联网，判 `ambiguous` 直接联网融合，
**两条路最终都到得了终点**，只是多花几秒。

### 待办

- **统一加载层**：PyMuPDF 支持 PDF/Markdown/文本，保留页码元数据用于答案溯源。
- **评测**：RAGAS 评测召回率、答案忠实度、相关性——「没量化 = 没做」。
- **实时类问题的前置门**：见上方「架构演进」，用不需要 LLM 的关键词筛省掉那 17s。

## 📝 已知限制

- **精排耗时长**：CPU 上单次检索约 5~6 秒，瓶颈在 CrossEncoder 对 500 字符 chunk 的推理。可行的优化方向：换更小的 reranker（如 `bge-reranker-base`，278M）、只取 chunk 前 N 个 token 送精排、或改用 GPU。
- **BM25 索引在内存**：启动时从 Chroma 全量拉取语料建索引。当前 15 个 chunk 无影响，大语料需改分页拉取或索引落盘。
- **中文停用词未处理**：已过滤空白和纯标点，但「什么」「的」这类高频功能词的 IDF 干扰仍在。改写节点（`rewrite.py`）从**查询侧**缓解了它——把疑问句压成名词短语，疑问词自然消失；但**语料侧**的功能词干扰仍在，可引入停用词表进一步优化。
