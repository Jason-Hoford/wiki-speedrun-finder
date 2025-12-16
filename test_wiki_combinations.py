import json
import time
from typing import Dict, List, Union
from bidirectional_walker import BidirectionalWalker

# Time limit in seconds
MAX_TIME_LIMIT = 300


def run_with_timeout(start_title: str, target_title: str, max_steps: int = 10, timeout: int = 60) -> Dict:
    """
    Run BidirectionalWalker and record time. If it exceeds timeout, mark as timeout.
    Returns dict with results including time, status, path_length, and found status.
    """
    result = {
        "start": start_title,
        "target": target_title,
        "found": False,
        "time_taken": 0.0,
        "path_length": 0,
        "status": "unknown",
        "path": []
    }
    
    start_time = time.perf_counter()
    
    try:
        # Run the bidirectional walker
        walker = BidirectionalWalker()
        path, status = walker.walk(
            start_title=start_title,
            target_title=target_title,
            max_steps=max_steps,  # Max depth per side
            beam_width=10, 
            verbose=False # Disable verbose for batch processing
        )
        
        elapsed = time.perf_counter() - start_time
        
        # Check if we exceeded time limit
        if elapsed > timeout:
            result["status"] = "timeout"
            result["found"] = False
            result["time_taken"] = timeout
            result["path_length"] = len(path) - 1 if path else 0
            result["path"] = path  # Still record the path even if timeout
        else:
            result["time_taken"] = elapsed
            result["status"] = status
            result["found"] = (status == "reached")
            result["path_length"] = len(path) - 1 if path else 0
            result["path"] = path
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        result["time_taken"] = min(elapsed, timeout)
        result["status"] = f"error: {str(e)[:100]}"
        result["found"] = False
    
    return result


def load_wiki_pages(json_file: str = "wiki_pages.json") -> Dict:
    """Load the wiki pages from JSON file. Supports entries with title/url or plain strings."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pages = data.get("pages", {})
        return pages
    except FileNotFoundError:
        print(f"[ERROR] File {json_file} not found. Please run collect_wiki_pages.py first.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {json_file}: {e}")
        return {}


def generate_combinations(pages: Dict, samples_per_category: int = 10) -> List[Dict]:
    """
    Generate all 9 combinations of source -> target categories.
    Returns list of test cases.
    """
    combinations = []
    
    def title_of(item: Union[str, Dict]) -> str:
        if isinstance(item, dict):
            return item.get("title") or item.get("name") or ""
        return str(item)
    
    def url_of(item: Union[str, Dict]) -> str:
        if isinstance(item, dict):
            return item.get("url", "")
        return ""
    
    categories = ["common", "semi_common", "random"]
    
    for source_cat in categories:
        for target_cat in categories:
            source_pages = pages.get(source_cat, [])
            target_pages = pages.get(target_cat, [])
            
            if not source_pages or not target_pages:
                print(f"[WARN] Missing pages for {source_cat} -> {target_cat}")
                continue
            
            # Generate sample combinations
            # Take samples from each list
            source_samples = source_pages[:min(samples_per_category, len(source_pages))]
            target_samples = target_pages[:min(samples_per_category, len(target_pages))]
            
            # Create combinations (avoid same page as source and target)
            for source in source_samples:
                for target in target_samples:
                    src_title = title_of(source)
                    tgt_title = title_of(target)
                    if not src_title or not tgt_title or src_title == tgt_title:
                        continue
                    combinations.append({
                        "source": src_title,
                        "target": tgt_title,
                        "source_url": url_of(source),
                        "target_url": url_of(target),
                        "source_category": source_cat,
                        "target_category": target_cat,
                        "combination": f"{source_cat} -> {target_cat}"
                    })
                    # Limit combinations per category pair
                    if len([c for c in combinations if c["combination"] == f"{source_cat} -> {target_cat}"]) >= samples_per_category:
                        break
                if len([c for c in combinations if c["combination"] == f"{source_cat} -> {target_cat}"]) >= samples_per_category:
                    break
    
    return combinations


def run_tests(combinations: List[Dict], max_steps: int = 30, timeout: int = 60) -> List[Dict]:
    """
    Run tests for all combinations.
    Returns list of results.
    """
    results = []
    total = len(combinations)
    
    print(f"\n[INFO] Running {total} test combinations...")
    print(f"[INFO] Time limit: {timeout} seconds per test")
    print(f"[INFO] Max steps: {max_steps}")
    print("=" * 80)
    
    for i, combo in enumerate(combinations, 1):
        print(f"\n[{i}/{total}] Testing: {combo['source']} -> {combo['target']}")
        print(f"         Category: {combo['combination']}")
        
        result = run_with_timeout(
            start_title=combo['source'],
            target_title=combo['target'],
            max_steps=max_steps,
            timeout=timeout
        )
        
        # Add combination info to result
        result["source_category"] = combo['source_category']
        result["target_category"] = combo['target_category']
        result["combination"] = combo['combination']
        
        results.append(result)
        
        # Print result
        status_icon = "[OK]" if result["found"] else "[FAIL]"
        print(f"         {status_icon} Found: {result['found']}, Time: {result['time_taken']:.2f}s, Path length: {result['path_length']}")
        
        # Progress update every 10 tests
        if i % 10 == 0:
            found_count = sum(1 for r in results if r["found"])
            print(f"\n[PROGRESS] {i}/{total} completed, {found_count} found so far")
    
    return results


def aggregate_results(results: List[Dict]) -> Dict:
    """
    Aggregate results by category combination.
    """
    aggregated = {}
    
    for result in results:
        combo = result["combination"]
        
        if combo not in aggregated:
            aggregated[combo] = {
                "total_tests": 0,
                "found_count": 0,
                "not_found_count": 0,
                "timeout_count": 0,
                "average_time": 0.0,
                "average_path_length": 0.0,
                "average_time_found": 0.0,
                "average_time_not_found": 0.0,
                "tests": []
            }
        
        agg = aggregated[combo]
        agg["total_tests"] += 1
        agg["tests"].append({
            "start": result["start"],
            "target": result["target"],
            "found": result["found"],
            "time_taken": result["time_taken"],
            "path_length": result["path_length"],
            "status": result["status"]
        })
        
        if result["found"]:
            agg["found_count"] += 1
        else:
            agg["not_found_count"] += 1
            if result["status"] == "timeout":
                agg["timeout_count"] += 1
    
    # Calculate averages
    for combo, agg in aggregated.items():
        if agg["total_tests"] > 0:
            total_time = sum(t["time_taken"] for t in agg["tests"])
            agg["average_time"] = total_time / agg["total_tests"]
            
            found_tests = [t for t in agg["tests"] if t["found"]]
            not_found_tests = [t for t in agg["tests"] if not t["found"]]
            
            if found_tests:
                agg["average_time_found"] = sum(t["time_taken"] for t in found_tests) / len(found_tests)
                agg["average_path_length"] = sum(t["path_length"] for t in found_tests) / len(found_tests)
            
            if not_found_tests:
                agg["average_time_not_found"] = sum(t["time_taken"] for t in not_found_tests) / len(not_found_tests)
    
    return aggregated


def main():
    print("=" * 80)
    print("=== Wikipedia Pathfinding Test Suite ===")
    print("=" * 80)
    
    # Configuration
    SAMPLES_PER_CATEGORY = 10  # Number of test pairs per combination
    MAX_STEPS = 100
    TIMEOUT = MAX_TIME_LIMIT
    
    # Load wiki pages
    print("\n[INFO] Loading wiki pages from wiki_pages.json...")
    pages = load_wiki_pages()
    
    if not pages:
        print("[ERROR] Failed to load wiki pages. Exiting.")
        return
    
    print(f"[INFO] Loaded pages:")
    print(f"  Common: {len(pages.get('common', []))}")
    print(f"  Semi-common: {len(pages.get('semi_common', []))}")
    print(f"  Random: {len(pages.get('random', []))}")
    
    # Generate combinations
    print(f"\n[INFO] Generating test combinations ({SAMPLES_PER_CATEGORY} per category pair)...")
    combinations = generate_combinations(pages, samples_per_category=SAMPLES_PER_CATEGORY)
    print(f"[INFO] Generated {len(combinations)} test combinations")
    
    # Show combination breakdown
    from collections import Counter
    combo_counts = Counter(c["combination"] for c in combinations)
    print("\n[INFO] Combinations breakdown:")
    for combo, count in sorted(combo_counts.items()):
        print(f"  {combo}: {count} tests")
    
    # Run tests
    overall_start = time.perf_counter()
    results = run_tests(combinations, max_steps=MAX_STEPS, timeout=TIMEOUT)
    overall_time = time.perf_counter() - overall_start
    
    # Aggregate results
    print("\n[INFO] Aggregating results...")
    aggregated = aggregate_results(results)
    
    # Create output structure
    output = {
        "metadata": {
            "test_run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(results),
            "samples_per_category": SAMPLES_PER_CATEGORY,
            "max_steps": MAX_STEPS,
            "timeout_seconds": TIMEOUT,
            "total_test_time": overall_time,
            "average_time_per_test": overall_time / len(results) if results else 0
        },
        "summary": {
            "total_found": sum(1 for r in results if r["found"]),
            "total_not_found": sum(1 for r in results if r["found"] == False),
            "total_timeouts": sum(1 for r in results if r["status"] == "timeout"),
            "overall_success_rate": sum(1 for r in results if r["found"]) / len(results) * 100 if results else 0
        },
        "by_combination": aggregated,
        "all_results": results
    }
    
    # Save to JSON
    output_file = "test_results.json"
    print(f"\n[INFO] Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("=== Test Results Summary ===")
    print("=" * 80)
    print(f"Total tests        : {len(results)}")
    print(f"Found              : {output['summary']['total_found']} ({output['summary']['overall_success_rate']:.1f}%)")
    print(f"Not found          : {output['summary']['total_not_found']}")
    print(f"Timeouts           : {output['summary']['total_timeouts']}")
    print(f"Total test time    : {overall_time:.2f}s ({overall_time/60:.2f} minutes)")
    print(f"Average per test   : {output['metadata']['average_time_per_test']:.2f}s")
    
    print("\n[INFO] Results by combination:")
    print("-" * 80)
    for combo, agg in sorted(aggregated.items()):
        success_rate = (agg["found_count"] / agg["total_tests"] * 100) if agg["total_tests"] > 0 else 0
        print(f"{combo:25s}: {agg['found_count']:3d}/{agg['total_tests']:3d} found ({success_rate:5.1f}%) | "
              f"Avg time: {agg['average_time']:5.2f}s | "
              f"Avg path: {agg['average_path_length']:.1f} links")
    
    print(f"\n[INFO] Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

