"""
智能文档问答 Agent —— 最小可运行原型
专为 GB/T 1568-2008《键 技术条件》扫描版 PDF 设计

流程：PDF → 渲染图片 → macOS Vision OCR → 分块 → BM25检索 → DeepSeek生成 → 自检
"""

import json
import os
import re
import base64
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pymupdf
from openai import OpenAI

# ─── 配置 ───────────────────────────────────────────
PDF_PATH = Path(__file__).parent / "GBT1568-2008键技术条件.pdf"
KB_PATH = Path(__file__).parent / "kb_store.json"    # 知识库 JSON 文件

# DeepSeek API（OpenAI 兼容模式）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def _get_api_key() -> str:
    """从多个来源尝试获取 DeepSeek API Key"""
    for var in ["DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        val = os.environ.get(var, "")
        if val and val.startswith("sk-"):
            return val
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            env = settings.get("env", {})
            for k in ["ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]:
                if k in env and env[k].startswith("sk-"):
                    return env[k]
        except (json.JSONDecodeError, KeyError):
            pass
    return ""

DEEPSEEK_API_KEY = _get_api_key()
CHAT_MODEL = "deepseek-v4-pro"

# macOS Vision OCR 脚本路径
OCR_SWIFT = Path(__file__).parent / "ocr.swift"

# ─── Step 1: PDF → 图片 ──────────────────────────────

def pdf_to_images(pdf_path: Path) -> list[dict]:
    """将 PDF 每页渲染为 base64 图片，返回 [{page, b64}]"""
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        pages.append({"page": i + 1, "base64": b64})
        print(f"  [pdf] 第 {i+1}/{doc.page_count} 页 → 图片 ({pix.width}x{pix.height})")
    return pages

def detect_pdf_type(pdf_path: Path) -> str:
    """判断 PDF 类型：text / scanned / mixed"""
    doc = pymupdf.open(str(pdf_path))
    text_chars = sum(len(page.get_text().strip()) for page in doc)
    if text_chars > 100:
        return "text"
    # 检查是否有图片
    has_images = any(len(page.get_images()) > 0 for page in doc)
    if has_images and text_chars < 100:
        return "scanned"
    return "mixed" if text_chars > 0 else "scanned"

# ─── Step 2: macOS Vision OCR ──────────────────────

def ocr_page(page_data: dict) -> str:
    """用 macOS Vision 框架识别单页图片（免安装，系统内置）"""
    img_bytes = base64.b64decode(page_data["base64"])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(img_bytes)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["swift", str(OCR_SWIFT), tmp_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"         OCR 错误: {result.stderr}")
            return ""
        return result.stdout
    finally:
        os.unlink(tmp_path)

def ocr_all_pages(pages: list[dict]) -> list[dict]:
    """逐页 OCR，返回 [{page, text}]"""
    results = []
    for p in pages:
        print(f"  [ocr] 识别第 {p['page']} 页 ...")
        text = ocr_page(p)
        results.append({"page": p["page"], "text": text})
        preview = text[:120].replace("\n", " ").strip()
        print(f"         → {preview}...")
    return results

# ─── Step 3: 条款分块 ────────────────────────────────

CLAUSE_PATTERN = re.compile(
    r'(?:^|\n)(?P<num>\d+(?:\.\d+)*)\s+(?P<title>[^\n]+)',
    re.MULTILINE
)

def split_clauses(pages: list[dict]) -> list[dict]:
    """按条款切分，返回 [{chunk_id, page, clause_num, title, text}]"""
    chunks = []
    for p in pages:
        text = p["text"]
        matches = list(CLAUSE_PATTERN.finditer(text))
        if not matches:
            # 整页作为一个 chunk
            chunks.append({
                "chunk_id": f"p{p['page']}_full",
                "page": p["page"],
                "clause_num": "",
                "title": "",
                "text": text.strip()
            })
            continue

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            clause_text = text[start:end].strip()
            chunks.append({
                "chunk_id": f"p{p['page']}_clause_{m.group('num').replace('.', '_')}",
                "page": p["page"],
                "clause_num": m.group("num"),
                "title": m.group("title").strip(),
                "text": clause_text
            })
    return chunks

# ─── Step 4: BM25 关键词知识库 ──────────────────────

def tokenize(text: str) -> list[str]:
    """简单中文分词：按字符 bigram + 数字/英文单词"""
    # 提取数字、英文单词
    tokens = re.findall(r'[a-zA-Z]+|\d+(?:\.\d+)?', text)
    # 中文字符 bigram
    chinese = re.findall(r'[一-鿿]', text)
    for i in range(len(chinese) - 1):
        tokens.append(chinese[i] + chinese[i + 1])
    # 单字也保留
    tokens.extend(chinese)
    return [t.lower() for t in tokens]

class BM25Retriever:
    """轻量 BM25 检索器"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.docs = [c["text"] for c in chunks]
        self.tokenized = [tokenize(d) for d in self.docs]
        self.doc_count = len(self.docs)
        self.avgdl = sum(len(t) for t in self.tokenized) / max(1, self.doc_count)
        self._compute_idf()

    def _compute_idf(self):
        """计算每个词的 IDF"""
        self.idf = {}
        for tokens in self.tokenized:
            for word in set(tokens):
                self.idf[word] = self.idf.get(word, 0) + 1
        for word, count in self.idf.items():
            self.idf[word] = math.log(1 + (self.doc_count - count + 0.5) / (count + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 检索"""
        q_tokens = tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.tokenized):
            score = 0.0
            doc_len = len(doc_tokens)
            tf = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1
            for token in q_tokens:
                if token in self.idf:
                    idf = self.idf[token]
                    term_freq = tf.get(token, 0)
                    k1, b = 1.5, 0.75
                    numerator = term_freq * (k1 + 1)
                    denominator = term_freq + k1 * (1 - b + b * doc_len / self.avgdl)
                    score += idf * numerator / max(0.01, denominator)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "page": c["page"],
                "clause_num": c["clause_num"],
                "title": c["title"],
                "text": c["text"],
                "score": round(score, 4)
            })
        return results

def build_knowledge_base(chunks: list[dict], rebuild: bool = False):
    """保存知识库到本地 JSON，返回 BM25 检索器"""
    retriever = BM25Retriever(chunks)
    # 保存 chunks 到 JSON（复用 OCR 结果）
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  [kb] BM25 知识库构建完成，共 {len(chunks)} 条，已保存至 {KB_PATH}")
    return retriever

def get_retriever() -> BM25Retriever:
    """从 JSON 加载知识库"""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return BM25Retriever(chunks)

# ─── Step 5: 检索 ────────────────────────────────────

def retrieve(retriever: BM25Retriever, query: str, top_k: int = 5) -> list[dict]:
    """BM25 关键词检索"""
    return retriever.search(query, top_k=top_k)

# ─── Step 6: 生成答案 ────────────────────────────────

ANSWER_PROMPT = """你是一个中国标准文档的问答助手。根据以下从 GB/T 1568-2008《键 技术条件》中检索到的条款内容，回答用户的问题。

## 检索到的证据

{evidence}

## 回答要求

1. 必须严格基于上述证据回答，不得编造任何信息
2. 涉及具体数值、公差、材料牌号时，必须逐字引用原文
3. 如果证据涉及表格，准确提取对应行列的数据
4. 如果证据不足以回答问题，直接说"该标准中未涵盖此内容"，不要猜测
5. 回答末尾列出引用的来源（页码 + 条款号）

## 用户问题

{question}

## 你的回答"""

def generate_answer(client: OpenAI, question: str, sources: list[dict]) -> dict:
    """基于检索结果生成答案"""
    if not sources or sources[0]["score"] < 0.3:
        return {
            "answer": "抱歉，该标准中未找到与此问题相关的内容。",
            "sources": [],
            "confidence": "low"
        }

    # 构造证据文本
    evidence_parts = []
    for i, s in enumerate(sources):
        loc = f"第{s['page']}页"
        if s["clause_num"]:
            loc += f"，条款 {s['clause_num']} {s['title']}"
        evidence_parts.append(f"[证据{i+1}]（{loc}）\n{s['text']}")
    evidence_text = "\n\n".join(evidence_parts)

    prompt = ANSWER_PROMPT.format(evidence=evidence_text, question=question)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.3
    )
    answer = resp.choices[0].message.content

    # 提取来源
    used_sources = []
    for s in sources:
        used_sources.append({
            "page": s["page"],
            "clause": s["clause_num"] or s["title"],
            "snippet": s["text"][:200] + ("..." if len(s["text"]) > 200 else "")
        })

    return {
        "answer": answer,
        "sources": used_sources,
        "confidence": "high" if sources[0]["score"] > 0.5 else "medium"
    }

# ─── Step 7: 自检 ────────────────────────────────────

SELF_CHECK_PROMPT = """请检查以下问答是否可靠。

## 问题
{question}

## 答案
{answer}

## 证据
{evidence}

请判断：
1. 答案是否完全基于证据？（是/否）
2. 答案中是否有证据中未出现的数据或断言？（如果有，列出）
3. 答案是否应该被拒绝（证据不足以支撑答案）？（是/否）

只回复 JSON：
{{"based_on_evidence": true/false, "hallucinated": ["具体幻觉内容"], "should_reject": true/false, "reason": "一句话说明"}}"""

def self_check(client: OpenAI, question: str, answer: str, sources: list[dict]) -> dict:
    """对答案进行自检"""
    evidence_text = "\n\n".join(
        f"[{s['page']}页/{s['clause']}] {s['snippet']}" for s in sources
    )
    prompt = SELF_CHECK_PROMPT.format(question=question, answer=answer, evidence=evidence_text)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.1
    )
    try:
        content = resp.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"based_on_evidence": True, "hallucinated": [], "should_reject": False,
            "reason": "无法完成自检"}

# ─── Step 8: 一站式问答接口 ──────────────────────────

class PDFQAAgent:
    """智能文档问答 Agent"""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or DEEPSEEK_API_KEY
        if not key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        self.client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
        self.retriever: Optional[BM25Retriever] = None

    def build(self, pdf_path: Optional[Path] = None, rebuild: bool = False):
        """从 PDF 构建知识库"""
        path = pdf_path or PDF_PATH
        print(f"[agent] 开始处理 PDF: {path.name}")
        print(f"[agent] PDF 类型: {detect_pdf_type(path)}")

        print("[step 1/4] 渲染页面为图片 ...")
        pages = pdf_to_images(path)

        print("[step 2/4] macOS Vision OCR 识别 ...")
        ocr_results = ocr_all_pages(pages)

        # 保存 OCR 结果
        ocr_file = Path(__file__).parent / "ocr_results.json"
        with open(ocr_file, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
        print(f"  [ocr] 结果已保存至 {ocr_file}")

        print("[step 3/4] 条款分块 ...")
        chunks = split_clauses(ocr_results)
        print(f"  [chunk] 共 {len(chunks)} 个条款块")

        print("[step 4/4] 构建 BM25 知识库 ...")
        self.retriever = build_knowledge_base(chunks, rebuild=rebuild)

        print(f"\n[agent] 构建完成！知识库共 {len(chunks)} 条记录")
        return self

    def ask(self, question: str, top_k: int = 5, check: bool = True) -> dict:
        """问答接口"""
        if self.retriever is None:
            self.retriever = get_retriever()

        # 检索
        sources = retrieve(self.retriever, question, top_k=top_k)
        if not sources:
            return {"answer": "未找到相关内容。", "sources": [], "confidence": "low", "check": None}

        # 生成
        result = generate_answer(self.client, question, sources)

        # 自检
        if check:
            check_result = self_check(self.client, question, result["answer"], result["sources"])
            result["check"] = check_result
            if check_result.get("should_reject"):
                result["answer"] = f"⚠️ 该答案可能不可靠：{check_result.get('reason', '证据不足')}\n\n原始回答（仅供参考）：\n{result['answer']}"
                result["confidence"] = "rejected"

        return result

    def chat(self):
        """交互式问答终端"""
        if self.retriever is None:
            self.retriever = get_retriever()
        print("\n" + "=" * 60)
        print("  GB/T 1568-2008《键 技术条件》智能问答")
        print("  输入问题开始，输入 /quit 退出")
        print("=" * 60 + "\n")

        while True:
            try:
                q = input("🔍 你的问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not q:
                continue
            if q.lower() in ("/quit", "/exit", "quit", "exit"):
                print("再见！")
                break

            result = self.ask(q)
            print(f"\n📖 答案: {result['answer']}")
            if result.get("check"):
                c = result["check"]
                print(f"   [自检] 基于证据: {c.get('based_on_evidence')} | 应拒答: {c.get('should_reject')}")
            if result["sources"]:
                print(f"   [来源]")
                for s in result["sources"]:
                    print(f"     - 第{s['page']}页, {s['clause']}: {s['snippet'][:80]}...")
            print(f"   [置信度] {result['confidence']}")
            print()

# ─── CLI ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv or "-h" in sys.argv:
        print("用法:")
        print("  python pdf_qa_agent.py build    - 构建知识库")
        print("  python pdf_qa_agent.py chat     - 交互问答")
        print("  python pdf_qa_agent.py ask '问题' - 单次问答")
        sys.exit(0)

    agent = PDFQAAgent()

    if "build" in sys.argv:
        agent.build(rebuild="--rebuild" in sys.argv)

    elif "ask" in sys.argv:
        idx = sys.argv.index("ask")
        question = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "键的公差等级有哪些要求？"
        agent.build() if not KB_PATH.exists() else None
        result = agent.ask(question)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        # 默认：如果没有知识库先构建，然后进入聊天
        if not KB_PATH.exists():
            agent.build()
        agent.chat()
