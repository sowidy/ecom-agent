from abc import ABC, abstractmethod
from dataclasses import dataclass

from mini_ecom_agent.rag.chunker import RagChunker


@dataclass
class RetrievedChunk:
    chunk: RagChunker
    score: float


class VectorBackend(ABC):
    @abstractmethod
    def upsert(
            self,
            chunks: list[RagChunker],
            vectors: list[list[float]],
            embedding_model: str,
    ) -> None:
        """全量重建索引（覆盖式）。

        每次调用都会清空既有数据再写入，避免脏数据。增量更新不在第 5 期范围。
        """

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """按余弦相似度（或等价度量）返回 Top-K 命中。"""

    @abstractmethod
    def size(self) -> int:
        """当前已索引的 chunk 数量。"""

    @abstractmethod
    def load(self) -> None:
        """从持久化存储加载索引；不存在时抛 FileNotFoundError。"""

    @abstractmethod
    def expected_embedding_model(self) -> str:
        """已持久化索引使用的 embedding 模型名（用于校验）。"""