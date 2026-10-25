import argparse
import sys
from pathlib import Path

# Windows 终端可能默认使用 GBK，无法输出部分 Unicode 字符。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mini_ecom_agent.config.settings import settings
from mini_ecom_agent.rag.backends import create_backend
from mini_ecom_agent.rag.chunker import chunk_markdown_dir
from mini_ecom_agent.rag.embedder import Embedder


def _validate_embedding_config() -> None:
    missing = []
    if not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    if not settings.embedding_base_url:
        missing.append("EMBEDDING_BASE_URL")
    if not settings.embedding_model:
        missing.append("EMBEDDING_MODEL")
    if missing:
        names = ", ".join(missing)
        raise SystemExit(
            f"❌ 缺少 Embedding 配置: {names}\n"
            "请在 .env 中单独配置支持 /embeddings 的 OpenAI 兼容服务。\n"
            "聊天服务 LLM_BASE_URL 与 Embedding 服务不能默认视为同一个接口。"
        )


def _build_backend(name: str):
    if name == "numpy":
        return create_backend(
            "numpy",
            index_path=PROJECT_ROOT / settings.kb_idx_path,
        ), str(PROJECT_ROOT / settings.kb_idx_path)
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=PROJECT_ROOT / settings.chroma_persistence_dir,
            collection_name=settings.chroma_collection,
        ), (
            f"{PROJECT_ROOT / settings.chroma_persistence_dir} "
            f"(collection={settings.chroma_collection})"
        )
    raise ValueError(f"未知后端: {name}")

def main():
    parser = argparse.ArgumentParser(description="构建知识库向量索引")
    parser.add_argument(
        "--backend",
        choices=["numpy", "chroma"],
        default=settings.rag_backend,
        help=f"向量后端（默认: {settings.rag_backend}）",
    )
    args = parser.parse_args()
    _validate_embedding_config()

    kb_dir = PROJECT_ROOT / settings.kb_dir

    if not kb_dir.exists():
        print(f"❌ 知识库目录不存在: {kb_dir}")
        sys.exit(1)

    backend, target_desc = _build_backend(args.backend)

    print("=" * 60)
    print("  并夕夕 · 知识库索引构建")
    print(f"  后端      : {args.backend}")
    print(f"  源目录    : {kb_dir}")
    print(f"  索引目标  : {target_desc}")
    print(f"  Embedding API: {settings.embedding_base_url}")
    print(f"  Embedding : {settings.embedding_model}")
    print("=" * 60)

    print("\n[1/3] 扫描并切分 markdown 文档...")
    chunks = chunk_markdown_dir(kb_dir)
    if not chunks:
        print("❌ 未发现任何文档，请检查 knowledge/ 目录")
        sys.exit(1)

    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
    for doc, n in by_doc.items():
        print(f"   - {doc}: {n} chunk")
    print(f"   合计 {len(chunks)} 个 chunk")

    print(f"\n[2/3] 调用 {settings.embedding_model} 批量向量化...")
    embedder = Embedder(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )
    try:
        vectors = embedder.encode([c.text for c in chunks])
    except Exception as exc:
        raise SystemExit(f"❌ Embedding 向量化失败: {exc}") from None
    dim = len(vectors[0]) if vectors else 0
    print(f"   完成，向量维度 = {dim}")

    print(f"\n[3/3] 写入 {args.backend} 索引...")
    backend.upsert(
        chunks=chunks,
        vectors=vectors,
        embedding_model=settings.embedding_model,
    )
    print(f"   已写入 {backend.size()} 条向量")

    print("\n🎉 索引构建完成。")


if __name__ == "__main__":
    main()
