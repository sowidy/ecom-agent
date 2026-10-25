
from pathlib import Path

from mini_ecom_agent.rag.backends.base import VectorBackend, RetrievedChunk
from mini_ecom_agent.rag.chunker import RagChunker

_EMBEDDING_MODEL_KEY = "embedding_model"


class ChromaBackend(VectorBackend):

    def __init__(self, persist_dir:Path, collection_name:str="ecom_kb"):
        self._persist_dir = Path(persist_dir)
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._size_cache = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb 未安装。请运行 `pip install chromadb` 或 "
                "`pip install -r requirements.txt`。"
            ) from e

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))

    def upsert(
        self,
        chunks: list[RagChunker],
        vectors: list[list[float]],
        embedding_model: str
    ):
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks 与 vectors 长度不一致: {len(chunks)} vs {len(vectors)}")
        self._ensure_client()
        # 全量重建：先删后建，避免 embedding 模型切换时的脏数据
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as e:
            pass

        # cosine 距离：score = 1 - distance，越大越相似
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={
                _EMBEDDING_MODEL_KEY: embedding_model,
                "hnsw:space": "cosine",
            }
        )

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"doc":c.doc, "section":c.section, "chunk_id":c.chunk_id} for c in chunks
            ]
        )

        self._embedding_model = embedding_model
        self._size_cache = len(chunks)

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if self._collection is None:
            self.load()
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        out: list[RetrievedChunk] = []
        for cid, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            chunk = RagChunker(
                chunk_id=meta.get("chunk_id", cid),
                doc=meta.get("doc", ""),
                section=meta.get("section", ""),
                text=doc_text,
            )
            score = 1.0 - float(dist)  # cosine distance → similarity
            out.append(RetrievedChunk(chunk=chunk, score=score))
        return out

    def size(self) -> int:
        if self._collection is None:
            try:
                self.load()
            except FileNotFoundError:
                return 0
        if self._size_cache is None:
            self._size_cache = self._collection.count()
        return self._size_cache

    def load(self) -> None:
        self._ensure_client()
        try:
            self._collection = self._client.get_collection(self._collection_name) # upsert时候创建,现在是直接获取
        except Exception as e:
            raise FileNotFoundError(
                f"Chroma collection '{self._collection_name}' 不存在于 "
                f"{self._persist_dir}。请先运行 "
                f"`python scripts/build_kb_index.py --backend chroma` 构建索引。"
            ) from e

        meta = self._collection.metadata or {}
        self._embedding_model = meta.get(_EMBEDDING_MODEL_KEY, "")
        self._size_cache = None  # 等需要时再 count

    def expected_embedding_model(self) -> str:
        if not self._embedding_model and self._collection is None:
            try:
                self.load()
            except FileNotFoundError:
                return ""
        return self._embedding_model
