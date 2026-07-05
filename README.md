# Cross-Border Credit Policy RAG

面向海外多国信贷政策、征信监管规则与金融合规文本的 RAG 检索评估管道。

本项目基于一个开源金融文档 RAG 评估框架二次重构，原框架主要用于企业财报检索；当前版本已经改造成更适合“跨国信贷合规政策库与征信数据中心”的系统骨架，重点解决法律条文 PDF 解析、法律层级保留、元数据硬过滤检索、严格防幻觉回答和检索质量评估。

## 项目目标

在印度 RBI、墨西哥央行、征信机构、乌干达监管机构等多国信贷合规文本中，实现可评估、可过滤、可追溯的 RAG 检索能力。

核心目标：

- 准确解析高度层级化的法律法规 PDF，尽量保留 Chapter、Article、Section 等结构。
- 按句子切块，保持当前实验最优基准：`chunk_size=500`，`overlap=0`。
- 给每个 chunk 注入国家、机构、文档类型、生效日期、法律层级等 metadata。
- 支持前端或 API 按国家、机构、文档类型等字段做前置硬过滤。
- 使用严格防幻觉 prompt，要求模型只基于检索到的政策切块回答，并强制标注出处。
- 保持原有 Grid Search 评估流程可运行，后续可用真实风控 QA 集重新评估 MRR、Recall@5、NDCG@5 等指标。

## 核心能力

### 1. Layout-Aware 法规 PDF 解析

文件：`src/parsing.py`

新增推荐解析器：

```python
from src.parsing import parse_layout_markdown

pages = parse_layout_markdown("data/policies/mexico_credit_policy.pdf")
```

解析优先级：

1. Marker：优先用于复杂 PDF 到 Markdown 的结构化转换。
2. Unstructured：使用 hi-res partition，适合复杂版式和表格。
3. PyMuPDF layout fallback：在本地缺少重依赖时仍可运行，按页面 block 顺序输出 Markdown。

解析器会尽量把以下法律标题转为 Markdown 标题：

- `Chapter 3`
- `Article 12`
- `Section 3`
- `Capítulo 2`
- `Artículo 8`
- `Sección 4`
- `第三章`
- `第十二条`
- `第三款`

### 2. 合规 Metadata Schema

文件：`src/models.py`

每个 chunk 的 metadata 保留原评估字段，同时新增合规字段：

```python
{
    "chunk_size": 500,
    "overlap": 0,
    "parser": "layout_markdown",
    "country": "Mexico",
    "institution": "Buro de Credito",
    "doc_type": "Compliance_Law",
    "effective_date": "2026-03-01",
    "legal_hierarchy": "Chapter 3 > Article 12 > Section 3"
}
```

字段说明：

| 字段 | 含义 | 示例 |
|---|---|---|
| `country` | 国家或司法辖区 | `India`, `Mexico`, `Uganda` |
| `institution` | 发文机构或征信中心 | `RBI`, `CIBIL`, `Buro de Credito` |
| `doc_type` | 文档类型 | `Compliance_Law`, `Credit_Reporting_Rule` |
| `effective_date` | 生效或发布时间 | `2026-03-01` |
| `legal_hierarchy` | 法律层级定位 | `Chapter 3 > Article 12 > Section 3` |

### 3. 句子切块与法律层级注入

文件：`src/chunking.py`

推荐入口：

```python
from src.parsing import parse_layout_markdown
from src.chunking import chunk_policy_document
from src.models import PolicyDocumentMetadata

pages = parse_layout_markdown("data/policies/mexico_credit_policy.pdf")

chunks = chunk_policy_document(
    pages,
    chunk_size=500,
    overlap=0,
    parser="layout_markdown",
    document_metadata=PolicyDocumentMetadata(
        country="Mexico",
        institution="Buro de Credito",
        doc_type="Compliance_Law",
        effective_date="2026-03-01",
    ),
)
```

默认策略：

- `chunker="sentence"`
- `chunk_size=500`
- `overlap=0`

切块时会自动扫描 Markdown 或普通文本中的法律标题，并为 chunk 注入 `legal_hierarchy`。如果下一页没有重复标题，系统会继承上一页最近的法律层级，避免跨页法条丢失上下文。

### 4. Metadata Hard Filter 检索

文件：

- `src/metadata_filter.py`
- `src/bm25_retrieval.py`
- `src/vector_store.py`
- `src/vector_retrieval.py`
- `src/grid_runner.py`

支持按 metadata 做硬过滤。过滤是前置约束，不是降低分数。

示例：

```python
metadata_filter = {
    "country": "Mexico",
    "institution": "Buro de Credito",
}
```

Grid Search 中使用：

```python
from src.grid_runner import run_phase2_grid

results = run_phase2_grid(
    configs_with_chunks={config.config_id: (config, chunks)},
    qa_by_config={config.config_id: qa_examples},
    metadata_filter={
        "country": "Mexico",
        "institution": "Buro de Credito",
    },
)
```

支持的过滤形式：

```python
{"country": "Mexico"}
{"country": ["Mexico", "India"]}
{"country": {"$eq": "Mexico"}}
{"institution": {"$in": ["RBI", "CIBIL"]}}
{"country": {"$ne": "Uganda"}}
```

### 5. 严格防幻觉 Prompt

文件：`src/prompting.py`

该模块提供合规场景专用 prompt，要求模型：

- 只能根据检索到的政策切块回答。
- 如果政策库没有收录对应规定，必须回答：`当前合规政策库暂未收录此条规定`。
- 回答合规限制、罚则或流程时必须附出处标签。

使用示例：

```python
from src.prompting import build_compliance_prompt

prompt = build_compliance_prompt(
    retrieved_chunks=chunks[:5],
    user_query="墨西哥征信机构在查询借款人信用报告前是否必须取得授权？",
)
```

生成的切块上下文会自动包含出处标签，例如：

```text
[Chunk 1] 出处标签：【Mexico-Buro de Credito-Compliance_Law-Chapter 3 > Article 12】
...
```

## 系统架构

```text
[PDF 法规文本]
      |
      v
src/parsing.py
layout-aware Markdown 解析
      |
      v
src/chunking.py
句子切块 + 法律层级抽取 + metadata 注入
      |
      v
data/chunks/{config}.jsonl
      |
      +--> src/embedding.py
      |    OpenAI Embedding + .npy 缓存
      |
      +--> src/qa_generator.py
      |    生成或加载风控 QA 测试集
      |
      v
src/grid_runner.py
BM25 / Vector / Hybrid 检索评估
      |
      v
outputs/{config}_results.json
      |
      v
src/visualizations.py
MRR / Recall / NDCG 等图表
```

## 目录结构

```text
src/
  parsing.py             # PDF 解析器，含 layout_markdown
  chunking.py            # 切块、法律层级抽取、metadata 注入
  models.py              # Pydantic 数据模型
  metadata_filter.py     # metadata 硬过滤逻辑
  bm25_retrieval.py      # BM25 检索
  vector_store.py        # FAISS 向量索引
  vector_retrieval.py    # 向量检索封装
  hybrid_retrieval.py    # BM25 + Vector 混合检索
  grid_runner.py         # Grid Search 评估
  metrics.py             # MRR、Recall@K、NDCG@K 等指标
  prompting.py           # 合规防幻觉 Prompt
  qa_generator.py        # QA 数据生成与持久化
  embedding.py           # OpenAI Embedding 与缓存

tests/
  test_chunking.py
  test_bm25_retrieval.py
  test_vector_store.py
  test_grid_runner.py
  test_prompting.py
  ...
```

## 快速开始

### 1. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要使用语义切块：

```bash
python -m spacy download en_core_web_md
```

如果需要最佳 PDF Markdown 解析质量，请安装并确认以下依赖可用：

```bash
pip install marker-pdf "unstructured[pdf]"
```

### 2. 配置 API Key

创建 `.env.local`：

```text
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=...
COHERE_API_KEY=...
```

`OPENAI_BASE_URL` 可选，适合代理或兼容 OpenAI API 的网关。

### 3. 准备政策 PDF

建议目录：

```text
data/policies/
  mexico_credit_policy.pdf
  india_rbi_credit_reporting.pdf
  uganda_credit_regulation.pdf
```

`data/` 默认被 `.gitignore` 忽略，不会误传敏感政策文本或测试数据。

### 4. 解析与切块

```python
from src.parsing import parse_layout_markdown
from src.chunking import chunk_policy_document
from src.models import ChunkingConfig, PolicyDocumentMetadata
from src.qa_generator import save_chunks, get_chunks_path

pdf_path = "data/policies/mexico_credit_policy.pdf"

pages = parse_layout_markdown(pdf_path)

config = ChunkingConfig(
    parser="layout_markdown",
    chunker="sentence",
    chunk_size=500,
    overlap=0,
)

chunks = chunk_policy_document(
    pages,
    chunk_size=500,
    overlap=0,
    parser="layout_markdown",
    document_metadata=PolicyDocumentMetadata(
        country="Mexico",
        institution="Buro de Credito",
        doc_type="Compliance_Law",
        effective_date="2026-03-01",
    ),
)

save_chunks(chunks, get_chunks_path(config.config_id))
```

### 5. 运行评估

你可以使用人工准备的 30 个风控真实 QA 对，也可以先用 `qa_generator.py` 生成测试 QA。

```python
from src.grid_runner import run_phase2_grid
from src.results_io import save_results

results = run_phase2_grid(
    configs_with_chunks={config.config_id: (config, chunks)},
    qa_by_config={config.config_id: qa_examples},
    embedding_models=["text-embedding-3-small", "text-embedding-3-large"],
    retrieval_methods=["bm25", "vector", "hybrid"],
)

save_results(results, f"outputs/{config.config_id}_results.json")
```

带国家和机构过滤：

```python
results = run_phase2_grid(
    configs_with_chunks={config.config_id: (config, chunks)},
    qa_by_config={config.config_id: qa_examples},
    metadata_filter={
        "country": "Mexico",
        "institution": "Buro de Credito",
    },
)
```

## 评估指标

项目保留原有评估框架，支持：

- `MRR`
- `Recall@1 / Recall@3 / Recall@5 / Recall@10`
- `Precision@K`
- `MAP`
- `NDCG@K`
- 平均检索耗时

后续替换为真实跨国信贷政策 PDF 和 30 个风控 QA 后，可以重新刷：

- `MRR`
- `Recall@5`
- `NDCG@5`

这些指标用于判断检索结果是否能把正确法条稳定放进前 5 个候选结果中。

## 测试

当前已验证的核心测试：

```bash
python3 -m pytest \
  tests/test_models.py \
  tests/test_chunking.py \
  tests/test_bm25_retrieval.py \
  tests/test_vector_store.py \
  tests/test_vector_retrieval.py \
  tests/test_grid_runner.py \
  tests/test_prompting.py \
  -q
```

本地验证结果：

```text
110 passed
```

## 当前状态

已完成：

- 将 PDF 解析升级为 layout-aware Markdown 入口。
- 扩展 chunk metadata schema，支持国家、机构、文档类型、生效日期、法律层级。
- 保留句子切块最佳基准：`500 chars / 0 overlap`。
- 切块阶段自动抽取并注入法律层级。
- BM25、FAISS、Hybrid、Grid Search 支持 metadata hard filter。
- 新增合规场景防幻觉 prompt。
- 保持原评估框架可运行。

待接入：

- 真实政策 PDF 数据集。
- 人工标注的 30 个风控 QA 对。
- 前端筛选条件，如国家、机构、文档类型、生效日期。
- 生产向量库，如 Chroma、Pinecone 或托管 FAISS 服务。
- 回答生成链路，将 `prompting.py` 与实际 LLM 调用封装成 API。

## 注意事项

- 本项目是合规检索和评估管道，不是法律意见生成系统。
- 生产环境中应保留完整出处、版本号、生效日期和原文链接。
- 对合规限制、罚则、准入流程等高风险问题，应默认采用“无出处不回答”策略。
- 如果检索结果没有覆盖用户问题，回答必须是：`当前合规政策库暂未收录此条规定`。

## 推荐下一步

1. 将印度、墨西哥、乌干达等政策 PDF 放入 `data/policies/`。
2. 为每份 PDF 准备文档级 metadata：国家、机构、文档类型、生效日期。
3. 运行 `parse_layout_markdown + chunk_policy_document` 生成 chunk。
4. 准备 30 个真实风控 QA 对，并绑定 ground-truth chunk id。
5. 跑 Grid Search，比较 BM25、Vector、Hybrid 的 MRR、Recall@5、NDCG@5。
6. 选出最佳检索配置后，再接入前端筛选和合规回答 API。
