import os
import pickle
from typing import Dict, Optional, List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    """
    Wrapper around a SentenceTransformer model with simple disk caching and batch processing.
    Cache grows without limits - all embeddings are kept.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_path: str = "embedding_cache.pkl",
    ):
        self.model_name = model_name
        self.cache_path = cache_path
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}

        self._load_cache()

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
                print(f"[INFO] Loaded {len(self.cache)} embeddings from cache.")
            except Exception as e:
                print(f"[WARN] Failed to load cache: {e}. Starting with empty cache.")
                self.cache = {}
        else:
            print("[INFO] No existing embedding cache found. Starting fresh.")

    def save_cache(self) -> None:
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
            print(f"[INFO] Saved {len(self.cache)} embeddings to cache.")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

    def get_or_compute(
        self,
        key: str,
        text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return the embedding for 'key'.
        If not cached, embed 'text' (or key itself if text is None), cache it, and return.
        Embeddings are L2-normalized by the model, so cosine similarity is just dot product.
        """
        if key in self.cache:
            return self.cache[key]

        if text is None:
            text = key

        vec = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        self.cache[key] = vec
        return vec

    def get_or_compute_batch(
        self,
        items: List[tuple],  # List of (key, text) tuples
    ) -> Dict[str, np.ndarray]:
        """
        Batch compute embeddings for multiple items.
        Returns dict mapping key -> embedding.
        Only computes embeddings for items not in cache.
        """
        # Separate cached and uncached items
        cached_results = {}
        uncached_items = []
        uncached_keys = []
        
        for key, text in items:
            if key in self.cache:
                cached_results[key] = self.cache[key]
            else:
                uncached_items.append(text if text is not None else key)
                uncached_keys.append(key)
        
        # Batch compute uncached embeddings
        if uncached_items:
            embeddings = self.model.encode(
                uncached_items, 
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32  # Process in batches for efficiency
            )
            
            # Store in cache and results
            for key, vec in zip(uncached_keys, embeddings):
                self.cache[key] = vec
                cached_results[key] = vec
        
        return cached_results
