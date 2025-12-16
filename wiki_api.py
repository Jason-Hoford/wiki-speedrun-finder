import requests
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

WIKI_REST_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"

# Global session for connection pooling (OPTIMIZATION: Connection pooling)
_session = None
_session_lock = threading.Lock()

# Global link cache instance (lazy loaded to avoid circular imports)
_link_cache = None

def get_session():
    """Get or create a global requests session with connection pooling."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                # Configure session for connection pooling and keep-alive
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=50,  # OPTIMIZED: Increased from 20 to 50 for better parallel performance
                    max_retries=3
                )
                _session.mount('http://', adapter)
                _session.mount('https://', adapter)
                _session.headers.update({"User-Agent": "WikiSpeedrunBot/0.1"})
    return _session

def get_thread_local_session():
    """Get a thread-local session for concurrent operations."""
    thread_local = threading.local()
    if not hasattr(thread_local, 'session'):
        thread_local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=3
        )
        thread_local.session.mount('http://', adapter)
        thread_local.session.mount('https://', adapter)
        thread_local.session.headers.update({"User-Agent": "WikiSpeedrunBot/0.1"})
    return thread_local.session

def get_link_cache():
    """Get or create the global link cache instance."""
    global _link_cache
    if _link_cache is None:
        from link_cache import LinkCache
        _link_cache = LinkCache()
    return _link_cache


def normalize_title(title: str) -> str:
    """
    Normalize a Wikipedia title for API usage:
    - strip whitespace
    - replace spaces with underscores
    """
    return title.strip().replace(" ", "_")


def get_wiki_url(title: str) -> str:
    """Get the full Wikipedia URL for a title."""
    norm_title = normalize_title(title)
    return WIKI_BASE_URL + norm_title.replace(" ", "_")


def get_page_summary(title: str) -> Tuple[str, Optional[str]]:
    """
    Fetch the canonical title and summary of a page using the REST API.
    Returns (canonical_title, summary_text or None).
    Uses connection pooling for better performance.
    """
    norm_title = normalize_title(title)
    url = WIKI_REST_SUMMARY_URL + norm_title
    session = get_session()
    resp = session.get(url, timeout=10)

    if resp.status_code != 200:
        return norm_title, None

    data = resp.json()
    canonical_title = data.get("title", norm_title)
    summary = data.get("extract")
    return canonical_title, summary


def get_outgoing_links(title: str, use_cache: bool = True) -> List[str]:
    """
    Use MediaWiki API to get ALL outgoing links (namespace 0 = article).
    Uses query API with pagination to ensure we get all links.
    Returns a list of linked page titles.
    Uses caching to avoid refetching links for pages we've seen before.
    """
    norm_title = normalize_title(title)
    
    # Check cache first
    if use_cache:
        cache = get_link_cache()
        cached_links = cache.get(norm_title)
        if cached_links is not None:
            return cached_links
    
    outgoing_titles = []
    continue_token = None

    while True:
        params = {
            "action": "query",
            "titles": norm_title,
            "prop": "links",
            "plnamespace": 0,  # Only article namespace
            "pllimit": 500,  # Max per request
            "format": "json",
            "redirects": 1,
        }
        
        if continue_token:
            params["plcontinue"] = continue_token

        try:
            session = get_session()
            resp = session.get(
                WIKI_API_URL, 
                params=params, 
                timeout=15
            )
            
            if resp.status_code != 200:
                break

            data = resp.json()
            query = data.get("query", {})
            pages = query.get("pages", {})
            
            # Get links from the page
            for page_id, page_data in pages.items():
                links = page_data.get("links", [])
                for link in links:
                    link_title = link.get("title")
                    if link_title:
                        outgoing_titles.append(link_title)
            
            # Check for continuation
            if "continue" in data:
                continue_token = data["continue"].get("plcontinue")
            else:
                break
                
        except Exception as e:
            print(f"[WARN] Error fetching links for '{title}': {e}")
            break

    # Cache the results
    if use_cache:
        cache = get_link_cache()
        cache.set(norm_title, outgoing_titles)

    return outgoing_titles


def get_incoming_links(title: str, limit: int = 500, use_cache: bool = True) -> List[str]:
    """
    Get list of pages that link TO the given title (incoming links/backlinks).
    Uses 'linkshere' property from MediaWiki API.
    
    Args:
        title: The page title to look up
        limit: Max links to return
        use_cache: Whether to use caching
        
    Returns:
        List of titles that link to this page
    """
    norm_title = normalize_title(title)
    
    # Use a specific cache key prefix for incoming links to differentiate from outgoing
    cache_key = f"INCOMING:{norm_title}"
    
    if use_cache:
        cache = get_link_cache()
        cached_links = cache.get(cache_key)
        if cached_links is not None:
            return cached_links

    incoming_titles = []
    continue_token = None
    
    # Safety limit to avoid infinite loops
    max_requests = 10 
    requests_made = 0

    while len(incoming_titles) < limit and requests_made < max_requests:
        params = {
            "action": "query",
            "titles": norm_title,
            "prop": "linkshere",
            "lhnamespace": 0,  # Only article namespace
            "lhlimit": min(500, limit - len(incoming_titles)),
            "format": "json",
            "redirects": 1,
        }
        
        if continue_token:
            params["lhcontinue"] = continue_token

        try:
            session = get_session()
            resp = session.get(
                WIKI_API_URL, 
                params=params, 
                timeout=15
            )
            requests_made += 1
            
            if resp.status_code != 200:
                break

            data = resp.json()
            query = data.get("query", {})
            pages = query.get("pages", {})
            
            # Extract links
            for page_id, page_data in pages.items():
                linkshere = page_data.get("linkshere", [])
                for link in linkshere:
                    link_title = link.get("title")
                    if link_title:
                        incoming_titles.append(link_title)
            
            # Check for continuation
            if "continue" in data:
                continue_token = data["continue"].get("lhcontinue")
            else:
                break
                
        except Exception as e:
            print(f"[WARN] Error fetching incoming links for '{title}': {e}")
            break

    # Cache the results
    if use_cache:
        cache = get_link_cache()
        cache.set(cache_key, incoming_titles)

    return incoming_titles


def get_page_summaries_batch(titles: List[str], max_workers: int = 15) -> dict:
    """
    Fetch summaries for multiple titles concurrently.
    Returns dict mapping title -> (canonical_title, summary)
    Uses thread-local sessions for connection pooling.
    """
    results = {}
    
    def fetch_summary(title):
        try:
            # Use thread-local session for connection pooling
            norm_title = normalize_title(title)
            url = WIKI_REST_SUMMARY_URL + norm_title
            session = get_thread_local_session()
            resp = session.get(url, timeout=10)
            
            if resp.status_code != 200:
                return title, (norm_title, None)
            
            data = resp.json()
            canonical_title = data.get("title", norm_title)
            summary = data.get("extract")
            return title, (canonical_title, summary)
        except Exception as e:
            return title, (title, None)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {executor.submit(fetch_summary, title): title for title in titles}
        for future in as_completed(future_to_title):
            title, result = future.result()
            results[title] = result
    
    return results


def get_outgoing_links_batch(titles: List[str], max_workers: int = 20) -> dict:
    """
    Fetch links for multiple titles concurrently.
    Returns dict mapping title -> list of links
    Uses thread-local sessions for connection pooling.
    """
    results = {}
    
    def fetch_links(title):
        try:
            # Use cached version which will use the session
            return title, get_outgoing_links(title, use_cache=True)
        except Exception as e:
            return title, []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {executor.submit(fetch_links, title): title for title in titles}
        for future in as_completed(future_to_title):
            title, links = future.result()
            results[title] = links
    
    return results
