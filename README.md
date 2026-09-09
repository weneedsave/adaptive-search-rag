# AdaptiveSearch-RAG 自适应检索问答系统

基于 LangGraph 的 Agentic RAG：LLM 路由判断问题来源，自动选择「本地知识库检索」或「联网搜索」，统一上下文后生成回答；支持 SQLite 会话持久化与完整单元测试。

## ✨ 项目简介

1. **查询路由**：LLM 结构化输出，判断问题属于「知识库内知识」还是「需要联网搜索」。
2. **检索增强**：ChromaDB 向量检索（bge-small-zh-v1.5 中文 embedding），召回本地知识库相关片段。
3. **联网兜底**：博查（Bocha）真实联网搜索，规整为统一上下文结构。
4. **会话持久化**：LangGraph SqliteSaver 保存对话状态，`thread_id` 区分会话，支持会话隔离。
5. **容错兜底**：路由非法值降级到本地检索；搜索失败降级为空上下文；空上下文不调用 LLM，减少幻觉。
6. **全量单元测试**：依赖注入 + mock，绝大多数测试无需真实 LLM / 向量库 / 网络。

## 🛠️ 技术栈

| 层 | 选型 |
| --- | --- |
| 编排 | LangGraph（StateGraph + conditional_edges） |
| 会话记忆 | langgraph-checkpoint-sqlite（SqliteSaver） |
| LLM | DeepSeek V4（`deepseek-v4-flash`，langchain-openai） |
| Embedding | BAAI/bge-small-zh-v1.5（sentence-transformers，本地离线） |
| 向量库 | ChromaDB（PersistentClient，余弦相似度） |
| 联网搜索 | 博查 Bocha AI Search API（httpx） |
| 后端语言 | Python 3.13 |

## 📁 目录结构

```
AdaptiveSearch-RAG/
├── src/adaptive_search_rag/
│   ├── state.py          # RAGState 状态定义（question/route/contexts/answer）
│   ├── llm.py            # DeepSeek LLM 构造（关 thinking）
│   ├── ingest.py         # 文档加载 + 分块 + 向量化入库
│   ├── graph.py          # LangGraph 图组装 + 条件路由 path_fn
│   ├── main.py           # CLI 交互入口、ask 问答函数
│   ├── nodes/
│   │   ├── route.py      # 路由节点：LLM 结构化输出 retrieve/search
│   │   ├── retrieve.py   # 本地知识库检索节点（依赖注入 retriever）
│   │   ├── search.py     # 联网搜索节点（失败降级）
│   │   └── generate.py   # 答案生成节点（空上下文兜底）
│   └── tools/
│       └── web_search.py # 博查 API 封装
├── data/                 # 知识库语料（Markdown）
├── tests/                # 单元测试（8 个测试文件，18 个用例）
├── scripts/              # 手动验证脚本
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

### 2. 交互式命令行问答

```bash
python -m adaptive_search_rag.main
```

输入 `exit` / `quit` 退出。

### 3. 功能验证脚本

自动验证基础问答、同 thread 会话持久化、不同 thread 会话隔离：

```bash
python scripts/verify_main.py
```

## 🧪 运行单元测试

```bash
pytest -v
```

> 绝大多数测试通过依赖注入假节点 / mock 掉网络，无需真实 LLM、向量库、网络，快速验证逻辑。
> 例外：`test_ingest_directory` 会真实加载本地 embedding 模型（约 20s）。

## 📌 核心流程

```
START → route_node（LLM 判断来源）
    ├─ route="retrieve" → retrieve_node 本地检索 → generate_node 生成答案
    └─ route="search"   → search_node 联网搜索   → generate_node 生成答案
generate_node → END
```

关键设计：

- **统一 contexts 结构**：retrieve 与 search 输出同构 `{"contexts": list[str]}`，generate 不关心上下文来源。
- **路由兜底**：route 为任意非 `"search"` 值（空串、拼写错误、非法值），`path_fn` 全部归入 retrieve 分支。
- **搜索降级**：博查调用失败 → 返回空 contexts，不中断图执行。
- **防幻觉**：contexts 为空时，generate 不调用 LLM，直接返回「根据现有资料无法回答」。

## ❓ 常见问题

1. Windows 控制台中文乱码：代码内置 `sys.stdout.reconfigure(encoding="utf-8")` 自动处理。
2. 清空所有对话记忆：删除项目里的 `checkpoints.sqlite` 文件。
3. 单元测试大多不依赖 `.env`（用假节点）；仅 `test_ingest_directory` 需本地有 embedding 模型缓存。

## 📈 拓展方向（阶段 2）

阶段 1 是「先路由」模式：路由先行判断来源，二选一。阶段 2 升级为「自适应检索」（CRAG）：

1. **统一加载层**：PyMuPDF 支持 PDF/Markdown/文本，保留页码元数据用于答案溯源。
2. **CRAG 自适应检索**：先检索 → 评估相关性 → 低质量才触发联网搜索。
3. **混合检索**：BM25（稀疏）+ 向量（稠密），再 rerank 重排。
4. **评测**：RAGAS 评测召回率、答案忠实度、相关性。

> 从「先路由」到「自适应检索」的演进，展示了检索策略从粗到精的思考过程。
