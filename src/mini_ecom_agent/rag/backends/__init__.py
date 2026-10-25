__all__ = ["RetrievedChunk", "VectorBackend", "create_backend"]

from mini_ecom_agent.rag.backends.base import RetrievedChunk, VectorBackend


def create_backend(name: str, **kwargs) -> VectorBackend:
    """工厂方法：根据名称创建对应后端。

    name: "numpy" | "chroma"
    kwargs: 后端特定参数（index_path / persist_dir / collection 等）
    """
    name = (name or "numpy").lower()
    if name == "numpy":
        pass
        # return NumpyBackend(index_path=kwargs["index_path"])
    if name == "chroma":
        from mini_ecom_agent.rag.backends.chroma_backend import ChromaBackend

        return ChromaBackend(
            persist_dir=kwargs["persist_dir"],
            collection_name=kwargs.get("collection_name", "ecom_kb"),
        )
    raise ValueError(f"未知的 RAG 后端: {name}（可选: numpy / chroma）")
