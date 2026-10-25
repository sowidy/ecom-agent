from typing import Iterable

from openai import NotFoundError, OpenAI


class Embedder:
    def __init__(
            self,
            api_key,
            base_url,
            model:str="text-embedding-3-small",
            batch_size:int=64
    ):
        if not api_key:
            raise ValueError("未配置 EMBEDDING_API_KEY")
        if not base_url:
            raise ValueError("未配置 EMBEDDING_BASE_URL")
        if not model:
            raise ValueError("未配置 EMBEDDING_MODEL")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model=model
        self._batch_size=batch_size

    @property
    def model(self):
        return self._model

    def encode(self, texts:Iterable[str]):
        """批量编码，自动按 batch_size 分批请求。"""
        texts = list(texts)
        if not texts:
            return []
        out:list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            try:
                resp = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except NotFoundError as exc:
                raise RuntimeError(
                    "Embedding API 返回 404。请确认 EMBEDDING_BASE_URL 支持 "
                    f"/embeddings，并且模型 {self._model!r} 存在。"
                ) from exc
            out.extend(item.embedding for item in resp.data)

        return out

    def encode_one(self, text:str):
        return self.encode([text])[0]
