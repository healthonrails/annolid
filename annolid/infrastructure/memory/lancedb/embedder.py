import os
import requests
from typing import List


class Embedder:
    """Abstraction for returning vector embeddings of text."""

    def __init__(
        self,
        provider: str = "jina",
        model: str = "jina-embeddings-v3",
        api_key: str = "",
        dimensions: int = 1024,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("ANNOLID_MEMORY_EMBEDDING_API_KEY", "")
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self.dimensions

        if self.provider == "jina":
            return self._embed_jina(text)
        elif self.provider == "openai":
            return self._embed_openai(text)
        else:
            # Fallback to zero vector when disabled or unsupported
            return [0.0] * self.dimensions

    def _embed_jina(self, text: str) -> List[float]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"model": self.model, "input": [text]}
            resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception:
            return [0.0] * self.dimensions

    def _embed_openai(self, text: str) -> List[float]:
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            resp = client.embeddings.create(
                input=[text],
                model=self.model,
            )
            return resp.data[0].embedding
        except Exception:
            return [0.0] * self.dimensions
