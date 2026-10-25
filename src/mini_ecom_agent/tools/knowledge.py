from typing import Optional

from mini_ecom_agent.config.settings import BASE_DIR, settings
from mini_ecom_agent.rag.backends import create_backend
from mini_ecom_agent.rag.embedder import Embedder
from mini_ecom_agent.rag.retriever import KnowledgeRetriever

_retriever: Optional[KnowledgeRetriever] = None

def _create_backend_from_settings():
    """根据 settings.rag_backend 创建对应后端实例。"""
    name = settings.rag_backend.lower()
    if name == "numpy":
        return create_backend(
            "numpy",
            index_path=BASE_DIR / settings.kb_idx_path,
        )
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=BASE_DIR / settings.chroma_persistence_dir,
            collection_name=settings.chroma_collection,
        )
    raise ValueError(
        f"未知的 RAG 后端: {settings.rag_backend}（可选: numpy / chroma）"
    )

def _get_retriever():
    global _retriever
    if _retriever is None:
        embedder = Embedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )
        backend = _create_backend_from_settings()
        _retriever = KnowledgeRetriever(embedder, backend)
        _retriever.load()

    return _retriever

def reset_retriever() -> None:
    """清空单例缓存（测试或切换后端时使用）。"""
    global _retriever
    _retriever = None


def search_knowledge(query: str, top_k: int = 3) -> dict:
    """检索退换货政策、配送说明、会员权益、FAQ 等知识库内容。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma",
          "query": str,
          "results": [
            {"doc": "...", "section": "...", "score": 0.83, "text": "..."},
            ...
          ],
          "error": "..."  # 仅失败时存在
        }
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空", "query": query, "results": []}

    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库初始化失败: {e}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    top_k = max(1, min(int(top_k or 3), 5))
    hits = retriever.search(query, top_k=top_k)

    return {
        "success": True,
        "backend": settings.rag_backend,
        "query": query,
        "results": [
            {
                "doc": h.chunk.doc,
                "section": h.chunk.section,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ],
    }
