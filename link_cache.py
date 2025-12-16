import os
import pickle
from typing import Dict, List, Optional

WIKI_LINKS_CACHE_PATH = "links_cache.pkl"


class LinkCache:
    """
    Cache for Wikipedia page links to avoid refetching.
    Similar to embedding cache but for link lists.
    Also caches link counts for quick existence checks.
    """
    def __init__(self, cache_path: str = WIKI_LINKS_CACHE_PATH):
        self.cache_path = cache_path
        self.cache: Dict[str, List[str]] = {}
        self.link_count_cache: Dict[str, int] = {}  # Cache of link counts for fast existence checks
        self._load_cache()

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    # Handle old cache format (just dict) and new format (tuple)
                    if isinstance(data, dict):
                        self.cache = data
                        # Build link count cache from existing data
                        self.link_count_cache = {title: len(links) for title, links in self.cache.items()}
                    else:
                        self.cache, self.link_count_cache = data
                print(f"[INFO] Loaded {len(self.cache)} link lists from cache.")
            except Exception as e:
                print(f"[WARN] Failed to load link cache: {e}. Starting with empty cache.")
                self.cache = {}
                self.link_count_cache = {}
        else:
            print("[INFO] No existing link cache found. Starting fresh.")

    def save_cache(self) -> None:
        try:
            with open(self.cache_path, "wb") as f:
                # Save both caches as a tuple
                pickle.dump((self.cache, self.link_count_cache), f)
            print(f"[INFO] Saved {len(self.cache)} link lists to cache.")
        except Exception as e:
            print(f"[WARN] Failed to save link cache: {e}")

    def get(self, title: str) -> Optional[List[str]]:
        """Get cached links for a title, or None if not cached."""
        return self.cache.get(title)

    def set(self, title: str, links: List[str]) -> None:
        """Cache links for a title."""
        self.cache[title] = links
        self.link_count_cache[title] = len(links)

    def has(self, title: str) -> bool:
        """Check if links are cached for a title."""
        return title in self.cache

    def has_links(self, title: str) -> Optional[bool]:
        """
        Check if a page has any links (without fetching them all).
        Returns None if not cached, True if has links, False if no links.
        """
        if title in self.link_count_cache:
            return self.link_count_cache[title] > 0
        return None

    def get_link_count(self, title: str) -> Optional[int]:
        """Get the number of links for a title, or None if not cached."""
        return self.link_count_cache.get(title)

