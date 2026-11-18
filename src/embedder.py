"""
Semantic Trigger Embedder using all-MiniLM-L6-v2 (384-d, CPU, L2-normalized).
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticTriggerEmbedder:
    """
    Computes normalized dense semantic embeddings on CPU using sentence-transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = 384

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embeds a single string or batch of strings into L2-normalized 384-d vector(s).
        """
        if isinstance(text, str):
            single = True
            texts = [text]
        else:
            single = False
            texts = text

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if single:
            return embeddings[0]
        return embeddings
