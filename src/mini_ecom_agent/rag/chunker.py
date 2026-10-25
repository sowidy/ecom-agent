import re
from dataclasses import asdict,dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200

@dataclass
class RagChunker:
    chunk_id: str
    doc:str
    section:str
    text:str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_markdown_dir(kb_dir:Path):
    """扫描目录下所有 .md 文件，逐一切分并汇总。"""
    chunks: list[RagChunker] = []
    for md_file in sorted(kb_dir.glob("*.md")):
        chunks.extend(_chunk_one_file(md_file))
    return chunks


def _split_by_h2(raw):
    """
    返回 [(section_title, section_body), ...]。
    """
    lines = raw.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []

    for line in lines:
        if line.startswith("# ") and not current_body and not sections:
            current_title = line[2:].strip() + " · 概览"
            continue
        if line.startswith("## "):
            if current_body:
                sections.append((current_title or "概览", current_body))
            current_title = line[3:].strip()
            current_body = []
            continue
        current_body.append(line)

    if current_body:
        sections.append((current_title or "概览", current_body))

    return [(t, "\n".join(b).strip()) for t, b in sections]


def _make_chunk(doc: str, section: str, text: str, idx: int, sub_idx: int) -> RagChunker:
    chunk_id = f"{doc}#{idx:02d}-{sub_idx:02d}"
    body = f"【{doc} · {section}】\n{text}"
    return RagChunker(chunk_id=chunk_id, doc=doc, section=section, text=body)


def _split_long(text: str) -> list[str]:
    """按段落贪心打包到 MAX_CHUNK_CHARS。"""
    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if buf and buf_len + len(p) + 2 > MAX_CHUNK_CHARS:
            pieces.append("\n\n".join(buf)) # 超过限制，加入最后list
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += len(p) + 2
    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def _chunk_one_file(md_path:Path) -> list[RagChunker]:
    raw =md_path.read_text(encoding="utf-8")
    doc_name = md_path.stem
    sections = _split_by_h2(raw)

    out:list[RagChunker] = []
    for idx, (section_title, section_body) in enumerate(sections):
        text = section_body.strip()
        if not text:
            continue
        if len(text) <= MAX_CHUNK_CHARS:
            out.append(_make_chunk(doc_name, section_title, text,idx,0))
            continue

        for sub_idx,piece in enumerate(_split_long(text)):
            out.append(_make_chunk(doc_name, section_title, piece,idx,sub_idx))
    return out
