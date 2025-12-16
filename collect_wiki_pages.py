import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import wiki_api
import re

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("[WARN] BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")

WIKI_REST_URL = "https://en.wikipedia.org/api/rest_v1"
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"


def get_session():
    """Get a requests session with connection pooling."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=3
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({"User-Agent": "WikiPageCollector/1.0"})
    return session


def get_popular_pages_from_wiki_page(limit: int = 200) -> List[str]:
    """
    Extract popular pages from Wikipedia:Popular_pages.
    This page contains curated lists of popular pages organized by categories.
    Returns list of page titles.
    """
    if not HAS_BS4:
        print("[WARN] BeautifulSoup4 required for extracting from Wikipedia:Popular_pages")
        return []
    
    session = get_session()
    titles = set()
    
    try:
        url = f"{WIKI_BASE_URL}Wikipedia:Popular_pages"
        resp = session.get(url, timeout=15)
        
        if resp.status_code != 200:
            print(f"[WARN] Failed to fetch Wikipedia:Popular_pages (status {resp.status_code})")
            return []
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Find all links in the content area
        # Popular pages are typically in lists, tables, or as regular links
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            print("[WARN] Could not find content area in Wikipedia:Popular_pages")
            return []
        
        # Extract links from various structures
        # Look for links in lists (ul, ol), tables, and paragraphs
        for link in content_div.find_all('a', href=True):
            href = link.get('href', '')
            
            # Only get article links (not categories, files, etc.)
            if href.startswith('/wiki/') and not href.startswith('/wiki/Category:') and not href.startswith('/wiki/File:') and not href.startswith('/wiki/Wikipedia:'):
                # Extract title from href
                title = href.replace('/wiki/', '').replace('_', ' ')
                # Decode URL encoding
                try:
                    from urllib.parse import unquote
                    title = unquote(title)
                except:
                    pass
                
                # Skip if it's the Popular_pages page itself or other meta pages
                if 'Popular_pages' not in title and 'Special:' not in title:
                    titles.add(title)
        
        print(f"[INFO] Extracted {len(titles)} unique pages from Wikipedia:Popular_pages")
        
    except Exception as e:
        print(f"[WARN] Error extracting popular pages from Wikipedia:Popular_pages: {e}")
    
    return list(titles)[:limit]


def get_most_viewed_pages(days_back: int = 7, limit: int = 100) -> List[str]:
    """
    Get most viewed Wikipedia pages from the last N days.
    Returns list of page titles.
    """
    session = get_session()
    all_titles = set()
    
    # Get most viewed pages for the last N days
    for i in range(days_back):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime("%Y/%m/%d")
        
        try:
            url = f"{WIKI_REST_URL}/page/most-viewed/en.wikipedia/all-access/{date_str}"
            resp = session.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])
                for item in items[:limit // days_back + 10]:  # Get some from each day
                    title = item.get("article")
                    if title:
                        all_titles.add(title)
        except Exception as e:
            print(f"[WARN] Error fetching most viewed for {date_str}: {e}")
            continue
    
    return list(all_titles)[:limit]


def get_random_pages(count: int = 100) -> List[str]:
    """
    Get random Wikipedia pages using the API.
    Can also use Special:Random as fallback, but API is preferred.
    Returns list of page titles.
    """
    session = get_session()
    titles = []
    
    # Use query API to get random pages (preferred method)
    batch_size = 50  # API limit is usually 50 per request
    batches = (count + batch_size - 1) // batch_size
    
    for batch in range(batches):
        try:
            params = {
                "action": "query",
                "list": "random",
                "rnnamespace": 0,  # Only articles
                "rnlimit": min(batch_size, count - len(titles)),
                "format": "json"
            }
            
            resp = session.get(WIKI_API_URL, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                random_items = data.get("query", {}).get("random", [])
                for item in random_items:
                    title = item.get("title")
                    if title:
                        titles.append(title)
        except Exception as e:
            print(f"[WARN] Error fetching random pages batch {batch + 1}: {e}")
            continue
    
    # If we still need more and API didn't provide enough, use Special:Random as fallback
    if len(titles) < count:
        print(f"[INFO] API provided {len(titles)}/{count} random pages. Using Special:Random for remaining...")
        try:
            # Special:Random redirects to a random page, we can follow redirects
            for _ in range(count - len(titles)):
                resp = session.get("https://en.wikipedia.org/wiki/Special:Random", 
                                 allow_redirects=True, timeout=10)
                if resp.status_code == 200:
                    # Extract title from final URL
                    final_url = resp.url
                    if '/wiki/' in final_url:
                        title = final_url.split('/wiki/')[-1].replace('_', ' ')
                        try:
                            from urllib.parse import unquote
                            title = unquote(title)
                        except:
                            pass
                        if title and title not in titles:
                            titles.append(title)
                time.sleep(0.1)  # Small delay to avoid rate limiting
        except Exception as e:
            print(f"[WARN] Error using Special:Random fallback: {e}")
    
    return titles[:count]


def get_popular_pages_from_featured(limit: int = 50) -> List[str]:
    """
    Get popular pages from Wikipedia's featured articles or popular categories.
    Returns list of page titles.
    """
    session = get_session()
    titles = set()
    
    # Get pages from "Featured articles" category
    try:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Featured articles",
            "cmnamespace": 0,
            "cmlimit": limit,
            "format": "json"
        }
        
        resp = session.get(WIKI_API_URL, params=params, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                title = member.get("title")
                if title and not title.startswith("Category:"):
                    titles.add(title)
    except Exception as e:
        print(f"[WARN] Error fetching featured articles: {e}")
    
    # Get pages from "Good articles" category (semi-common)
    try:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Good articles",
            "cmnamespace": 0,
            "cmlimit": limit,
            "format": "json"
        }
        
        resp = session.get(WIKI_API_URL, params=params, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            members = data.get("query", {}).get("categorymembers", [])
            for member in members[:limit]:
                title = member.get("title")
                if title and not title.startswith("Category:"):
                    titles.add(title)
    except Exception as e:
        print(f"[WARN] Error fetching good articles: {e}")
    
    return list(titles)[:limit]


def has_links(title: str) -> bool:
    """
    Quick check if a page has any outgoing links.
    Returns True if page has at least one link, False otherwise.
    """
    try:
        links = wiki_api.get_outgoing_links(title, use_cache=True)
        return len(links) > 0
    except Exception as e:
        return False


def has_links_batch(titles: List[str], max_workers: int = 20) -> Dict[str, bool]:
    """
    Check multiple pages for links in parallel.
    MUCH faster than checking sequentially.
    Returns dict mapping title -> has_links (bool)
    """
    results = {}
    
    def check_page(title):
        try:
            links = wiki_api.get_outgoing_links(title, use_cache=True)
            return title, len(links) > 0
        except Exception as e:
            return title, False
    
    # Use ThreadPoolExecutor for parallel checking
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {executor.submit(check_page, title): title for title in titles}
        for future in as_completed(future_to_title):
            title, has_links_result = future.result()
            results[title] = has_links_result
    
    return results


def filter_pages_with_links(pages: List[str], target_count: int, category_name: str, fetch_more_func=None) -> List[str]:
    """
    Filter pages to only keep those with links.
    If we don't have enough, fetch more pages until we reach target_count.
    OPTIMIZED: Uses parallel batch processing for 10-20x speedup.
    
    Args:
        pages: List of page titles to check
        target_count: Number of pages with links we want
        category_name: Name of category (for logging)
        fetch_more_func: Optional function to fetch more pages if needed
    
    Returns:
        List of pages that have links
    """
    valid_pages = []
    checked = 0
    skipped = 0
    
    print(f"[INFO] Checking {len(pages)} {category_name} pages for links (parallel batch processing)...")
    
    # OPTIMIZATION: Check pages in batches using parallel processing
    batch_size = 50  # Check 50 pages at a time
    
    for i in range(0, len(pages), batch_size):
        if len(valid_pages) >= target_count:
            break
        
        batch = pages[i:i+batch_size]
        
        # Check entire batch in parallel
        batch_results = has_links_batch(batch, max_workers=20)
        
        # Add valid pages from batch
        for title in batch:
            checked += 1
            if batch_results.get(title, False):
                valid_pages.append(title)
                if len(valid_pages) >= target_count:
                    break
            else:
                skipped += 1
        
        # Progress update
        print(f"  Checked {checked}, found {len(valid_pages)} with links, skipped {skipped}")
    
    # If we don't have enough and we have a fetch function, get more
    if len(valid_pages) < target_count and fetch_more_func:
        print(f"[INFO] Only found {len(valid_pages)}/{target_count} pages with links. Fetching more...")
        additional_needed = target_count - len(valid_pages)
        max_attempts = additional_needed * 3  # Try up to 3x more than needed
        
        attempts = 0
        while len(valid_pages) < target_count and attempts < max_attempts:
            # Fetch a batch of new pages
            fetch_count = min(100, additional_needed * 2)  # Fetch more at once since we're batching
            new_pages = fetch_more_func(count=fetch_count)
            
            # Remove duplicates
            new_pages = [p for p in new_pages if p not in valid_pages]
            
            if not new_pages:
                print("[WARN] No new pages fetched, stopping.")
                break
            
            # Check new pages in parallel batches
            for i in range(0, len(new_pages), batch_size):
                if len(valid_pages) >= target_count:
                    break
                
                batch = new_pages[i:i+batch_size]
                batch_results = has_links_batch(batch, max_workers=20)
                
                for title in batch:
                    checked += 1
                    attempts += 1
                    
                    if batch_results.get(title, False):
                        valid_pages.append(title)
                        if len(valid_pages) >= target_count:
                            break
                    else:
                        skipped += 1
                    
                    if attempts >= max_attempts:
                        break
                
                print(f"  Checked {checked}, found {len(valid_pages)} with links, skipped {skipped}")
                
                if attempts >= max_attempts:
                    break
    
    print(f"[INFO] Found {len(valid_pages)} {category_name} pages with links (checked {checked}, skipped {skipped})")
    return valid_pages[:target_count]


def to_entry(title: str) -> Dict[str, str]:
    """Return a consistent page entry with title and full URL."""
    return {"title": title, "url": wiki_api.get_wiki_url(title)}


def categorize_pages(common_pages: List[str], semi_common_pages: List[str], random_pages: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Organize pages into categories as list of entries {title, url}.
    """
    return {
        "common": [to_entry(t) for t in common_pages],
        "semi_common": [to_entry(t) for t in semi_common_pages],
        "random": [to_entry(t) for t in random_pages],
    }


def main():
    print("=" * 80)
    print("=== Wikipedia Page Collector ===")
    print("=" * 80)
    
    # Configuration
    COMMON_COUNT = 200      # Most viewed/popular pages
    SEMI_COMMON_COUNT = 200  # Featured/Good articles
    RANDOM_COUNT = 200      # Random pages
    
    print(f"\n[INFO] Collecting {COMMON_COUNT} common pages (most viewed)...")
    common_start = time.perf_counter()
    
    # Get popular pages from Wikipedia:Popular_pages (best source)
    print("[INFO] Extracting popular pages from Wikipedia:Popular_pages...")
    popular_wiki_pages = get_popular_pages_from_wiki_page(limit=COMMON_COUNT * 2)
    
    # Also get from API most viewed
    common_pages_raw = get_most_viewed_pages(days_back=7, limit=COMMON_COUNT * 2)
    
    # Also get some from featured articles
    featured_pages = get_popular_pages_from_featured(limit=50)
    
    # Combine all sources
    common_pages_raw = list(set(popular_wiki_pages + common_pages_raw + featured_pages))
    print(f"[INFO] Combined {len(common_pages_raw)} pages from all sources")
    
    # Filter to only pages with links
    common_pages = filter_pages_with_links(
        common_pages_raw,
        target_count=COMMON_COUNT,
        category_name="common",
        fetch_more_func=lambda count: get_most_viewed_pages(days_back=30, limit=count)
    )
    common_time = time.perf_counter() - common_start
    print(f"[INFO] Collected {len(common_pages)} common pages with links in {common_time:.2f}s")
    
    print(f"\n[INFO] Collecting {SEMI_COMMON_COUNT} semi-common pages (featured/good articles)...")
    semi_start = time.perf_counter()
    semi_common_pages_raw = get_popular_pages_from_featured(limit=SEMI_COMMON_COUNT * 2)
    # If we need more, get some from most viewed that aren't in common
    if len(semi_common_pages_raw) < SEMI_COMMON_COUNT:
        more_viewed = get_most_viewed_pages(days_back=30, limit=SEMI_COMMON_COUNT * 2)
        for page in more_viewed:
            if page not in common_pages and page not in semi_common_pages_raw:
                semi_common_pages_raw.append(page)
    
    # Filter to only pages with links
    semi_common_pages = filter_pages_with_links(
        semi_common_pages_raw,
        target_count=SEMI_COMMON_COUNT,
        category_name="semi-common",
        fetch_more_func=lambda count: get_popular_pages_from_featured(limit=count)
    )
    semi_time = time.perf_counter() - semi_start
    print(f"[INFO] Collected {len(semi_common_pages)} semi-common pages with links in {semi_time:.2f}s")
    
    print(f"\n[INFO] Collecting {RANDOM_COUNT} random pages...")
    random_start = time.perf_counter()
    random_pages_raw = get_random_pages(count=RANDOM_COUNT * 2)  # Get more to account for filtering
    
    # Filter to only pages with links
    random_pages = filter_pages_with_links(
        random_pages_raw,
        target_count=RANDOM_COUNT,
        category_name="random",
        fetch_more_func=lambda count: get_random_pages(count=count)
    )
    random_time = time.perf_counter() - random_start
    print(f"[INFO] Collected {len(random_pages)} random pages with links in {random_time:.2f}s")
    
    # Organize into categories
    categorized = categorize_pages(common_pages, semi_common_pages, random_pages)
    
    # Add metadata
    output = {
        "metadata": {
            "collected_at": datetime.now().isoformat(),
            "common_count": len(categorized["common"]),
            "semi_common_count": len(categorized["semi_common"]),
            "random_count": len(categorized["random"]),
            "total_count": len(categorized["common"]) + len(categorized["semi_common"]) + len(categorized["random"])
        },
        "pages": categorized
    }
    
    # Save to JSON file
    output_file = "wiki_pages.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("=== Collection Complete ===")
    print("=" * 80)
    print(f"Common pages    : {len(categorized['common'])}")
    print(f"Semi-common pages: {len(categorized['semi_common'])}")
    print(f"Random pages    : {len(categorized['random'])}")
    print(f"Total pages     : {output['metadata']['total_count']}")
    print(f"\nSaved to: {output_file}")
    print("=" * 80)
    
    # Show some examples
    print("\nSample pages from each category:")
    print(f"\nCommon (first 5):")
    for i, page in enumerate(categorized['common'][:5], 1):
        print(f"  {i}. {page['title']} ({page['url']})")
    print(f"\nSemi-common (first 5):")
    for i, page in enumerate(categorized['semi_common'][:5], 1):
        print(f"  {i}. {page['title']} ({page['url']})")
    print(f"\nRandom (first 5):")
    for i, page in enumerate(categorized['random'][:5], 1):
        print(f"  {i}. {page['title']} ({page['url']})")


if __name__ == "__main__":
    main()

"""
python generate_graphs.py
"""