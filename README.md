# 智能文档问答 Agent

针对 GB/T 1568-2008《键 技术条件》扫描版 PDF 的最小可运行原型。

## 最终架构

```
PDF → PyMuPDF渲染 → macOS Vision OCR(swift) → 条款分块 → BM25检索 → DeepSeek生成 → 自检
```

| 组件 | 选型 | 说明 |
|------|------|------|
| PDF 渲染 | PyMuPDF | 300 DPI 渲染为图片 |
| OCR | macOS Vision 框架 | 系统内置，零安装，中文识别好 |
| 条款分块 | 正则匹配 | 识别 "3.1""4.2.1" 等条款编号 |
| 检索 | BM25（自实现） | 零依赖，关键词 + bigram 分词 |
| 答案生成 | DeepSeek v4-pro | 基于检索证据生成，要求逐字引用 |
| 自检 | DeepSeek v4-pro | 证据一致性、幻觉检测、拒答判定 |
| 知识库存储 | JSON 文件 | 轻量持久化 |

## 文件结构

```
pdf_qa_agent.py      # 主程序
ocr.swift            # macOS 原生 OCR 脚本
ocr_results.json     # OCR 提取结果（缓存，构建时生成）
kb_store.json        # BM25 知识库（构建时生成）
```

## 环境要求

- macOS 13+（使用 Vision 框架做 OCR）
- Python 3.12+
- DeepSeek API Key

## 安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install pymupdf openai
```

> 注意：本原型无需安装 Tesseract、PyTorch 或 ChromaDB，所有检索和 OCR 均使用系统内置能力或自实现。

## 使用方式

### 1. 构建知识库

```bash
python pdf_qa_agent.py build
```

对 PDF 逐页 OCR → 条款分块 → 构建 BM25 索引 → 保存至 `kb_store.json`。  
OCR 结果同时缓存至 `ocr_results.json`，重复构建时可复用。

强制重建：

```bash
python pdf_qa_agent.py build --rebuild
```

### 2. 单次问答

```bash
python pdf_qa_agent.py ask "键的抗拉强度要求是多少？"
```

输出 JSON，包含答案、来源（页码 + 条款号）、置信度、自检结果。

### 3. 交互式问答

```bash
python pdf_qa_agent.py chat
```

进入交互终端，输入问题即可获得回答，输入 `/quit` 退出。

## 问答流程

1. **检索** —— BM25 对用户问题进行关键词匹配，返回 Top-5 相关条款
2. **生成** —— 将检索到的条款作为证据，交由 DeepSeek 生成答案，要求逐字引用原文并标注来源
3. **自检** —— 对答案进行证据一致性验证：是否有依据、是否幻觉、是否需要拒答

## 测试示例

| 问题 | 预期 |
|------|------|
| 键的材料强度要求是什么？ | 返回 "抗拉强度 ≥ 590 MPa"，来源 3.1 |
| 键的公差等级有哪些要求？ | 返回平行度公差和楔键角度公差，来源 3.5、3.6 |
| 这个标准适用什么螺丝？ | 指出标准适用键而非螺丝（拒答） |
| 键的包装要求是什么？ | 返回 5.1~5.4 的包装和防锈要求 |
# ai-chat-bot
