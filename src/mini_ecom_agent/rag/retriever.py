from mini_ecom_agent.rag.backends.base import VectorBackend, RetrievedChunk
from mini_ecom_agent.rag.embedder import Embedder

__all__ = ["KnowledgeRetriever", "RetrievedChunk"]


class KnowledgeRetriever:
    """对上层暴露统一接口，对下委托给具体 backend。"""

    def __init__(self, embedder: Embedder, backend: VectorBackend):
        self._embedder = embedder
        self._backend = backend
        self._loaded = False

    @property
    def backend(self):
        return self._backend

    @property
    def size(self):
        return self._backend.size

    def load(self):
        """
        加载向量索引，并确认索引使用的嵌入模型与当前模型一致
        """
        if self._loaded:
            return
        self._backend.load() # 调用具体向量后端的加载方法。

        expected = self._backend.expected_embedding_model() # 读取索引保存的嵌入模型名称
        if expected and expected != self._embedder.model:
            raise ValueError(
                f"索引模型({expected}) 与当前 Embedder 模型"
                f"({self._embedder.model}) 不一致，请重建索引。"
            )
        self._loaded = True

    def search(self,query:str,top_k:int=3):
        if not self._loaded:
            self.load()
        q_vec = self._embedder.encode_one(query)
        return self._backend.search(q_vec, top_k=top_k)