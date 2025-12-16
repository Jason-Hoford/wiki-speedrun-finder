from typing import List, Tuple, Dict, Set, Optional
import time
import numpy as np

from embeddings import EmbeddingStore
import wiki_api
from link_cache import LinkCache
from safe_print import safe_print


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cosine_similarity_vectorized(embeddings_matrix: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    return np.dot(embeddings_matrix, target_vec)


class BidirectionalWalker:
    def __init__(self):
        self.embedder = EmbeddingStore()
        self.link_cache = LinkCache()
        self.verbose = True

    def walk(
        self,
        start_title: str,
        target_title: str,
        max_steps: int = 10,  # Max depth per side (so total path length up to 2*max_steps)
        beam_width: int = 10,  # Keeping top K paths per side
        verbose: bool = True,
        return_history: bool = False
    ) -> Tuple[List[str], str, Optional[List[Dict]]]:
        """
        Perform bidirectional semantic beam search.
        
        Args:
            start_title: Starting page title
            target_title: Target page title
            max_steps: Maximum depth to search from EACH side
            beam_width: Number of active paths to keep per side
            verbose: Whether to print progress
            return_history: Whether to return search history for visualization
            
        Returns:
            (path, status) or (path, status, history) if return_history=True
        """
        self.verbose = verbose
        
        # 1. Initialize Start and Target
        start_canonical, _ = wiki_api.get_page_summary(start_title)
        target_canonical, _ = wiki_api.get_page_summary(target_title)
        
        start_vec = self.embedder.get_or_compute(start_canonical, start_canonical)
        target_vec = self.embedder.get_or_compute(target_canonical, target_canonical)
        
        # Frontiers: Map of {current_title: path_list}
        # Path list includes the current_title
        forward_frontier: Dict[str, List[str]] = {start_canonical: [start_canonical]}
        backward_frontier: Dict[str, List[str]] = {target_canonical: [target_canonical]}
        
        # Visited sets to prevent cycles and redundant work
        # For backward search, visited means "we have found a path FROM this node TO target"
        visited_forward: Set[str] = {start_canonical}
        visited_backward: Set[str] = {target_canonical}
        
        start_time = time.perf_counter()
        
        if self.verbose:
            safe_print("=" * 80)
            safe_print("=== Bidirectional Semantic Beam Search ===")
            safe_print(f"Start : {start_canonical}")
            safe_print(f"Target: {target_canonical}")
            safe_print(f"Width : {beam_width}")
            safe_print("=" * 80)

        # Check direct link first
        if start_canonical == target_canonical:
             if return_history:
                 return [start_canonical], "reached", []
             return [start_canonical], "reached"

        # Visited maps: title -> path_to_this_node
        visited_forward: Dict[str, List[str]] = {start_canonical: [start_canonical]}
        visited_backward: Dict[str, List[str]] = {target_canonical: [target_canonical]}
        
        history: List[Dict] = []

        for step in range(1, max_steps + 1):
            step_start_time = time.perf_counter()
            if self.verbose:
                safe_print(f"\n[STEP {step}] Forward Frontier: {len(forward_frontier)}, Backward Frontier: {len(backward_frontier)}")

            # --- 2. Expand Forward (Start -> Target) ---
            next_forward_frontier: Dict[str, List[str]] = {}
            current_history_step = {
                "step": step,
                "direction": "forward",
                "frontier": list(forward_frontier.keys()),
                "candidates": [], # List of (child, parent)
                "selected": []
            }
            
            if self.verbose:
                safe_print(f"  [>] Expanding Forward ({len(forward_frontier)} nodes)...")
            
            # Batch fetch outgoing
            forward_titles = list(forward_frontier.keys())
            all_outgoing = wiki_api.get_outgoing_links_batch(forward_titles, max_workers=20)
            
            # 1. Collect all valid edges and check intersections
            all_edges = []
            unique_children = set()
            
            for parent, path in forward_frontier.items():
                links = all_outgoing.get(parent, [])
                for link in links:
                    if link not in visited_forward:
                        # CHECK INTERSECTION with FULL BACKWARD HISTORY
                        if link in visited_backward:
                            # Found connection!
                            path_from_start = path + [link]
                            path_from_target_reversed = visited_backward[link] # [link, parent, ..., Target] ?
                            # My backward path convention in frontier was [link, ..., Target]
                            # Let's assume visited_backward stores it same way.
                            
                            # However, we need to be careful not to duplicate 'link'
                            # path_from_start ends with link.
                            # visited_backward[link] starts with link.
                            
                            full_path = path_from_start[:-1] + visited_backward[link]
                            
                            elapsed = time.perf_counter() - start_time
                            if self.verbose:
                                safe_print(f"\n[SUCCESS] Meeting point found (Forward step): '{link}'")
                                safe_print(f"Total time: {elapsed:.2f}s")
                            self.embedder.save_cache()
                            self.link_cache.save_cache()
                            
                            if return_history:
                                return full_path, "reached", history
                            return full_path, "reached"
                        
                        all_edges.append((parent, link))
                        unique_children.add(link)
                        if return_history:
                            current_history_step["candidates"].append((link, parent))
            
            if not unique_children:
                if self.verbose: safe_print("  [>] Forward search dead end.")
                forward_frontier = {}
            else:
                # 2. Score children
                child_list = list(unique_children)
                embeddings = self.embedder.get_or_compute_batch([(t, t) for t in child_list])
                child_vecs = np.array([embeddings[t] for t in child_list])
                scores = cosine_similarity_vectorized(child_vecs, target_vec)
                child_scores = {title: scores[i] for i, title in enumerate(child_list)}
                
                # 3. Select top K
                best_children = sorted(child_scores.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                best_children_set = set(t for t, _ in best_children)
                
                if return_history:
                    current_history_step["selected"] = list(best_children_set)
                
                # 4. Build next frontier
                for parent, link in all_edges:
                    if link in best_children_set and link not in next_forward_frontier:
                        new_path = forward_frontier[parent] + [link]
                        next_forward_frontier[link] = new_path
                        visited_forward[link] = new_path  # Save to history
                
                forward_frontier = next_forward_frontier
                
                if self.verbose:
                    best_score = best_children[0][1] if best_children else 0
                    safe_print(f"  [>] Forward best score: {best_score:.4f} ('{best_children[0][0]}' if any)")

            if return_history:
                history.append(current_history_step)
            
            # --- 3. Expand Backward (Target -> Start) ---
            next_backward_frontier: Dict[str, List[str]] = {}
            current_history_step_back = {
                "step": step,
                "direction": "backward",
                "frontier": list(backward_frontier.keys()),
                "candidates": [], 
                "selected": []
            }
            
            if self.verbose:
                safe_print(f"  [<] Expanding Backward ({len(backward_frontier)} nodes)...")
                
            backward_titles = list(backward_frontier.keys())
            
            # Parallel fetch incoming
            from concurrent.futures import ThreadPoolExecutor, as_completed
            incoming_map = {} 
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_title = {
                    executor.submit(wiki_api.get_incoming_links, t, limit=200): t 
                    for t in backward_titles
                }
                for future in as_completed(future_to_title):
                    title = future_to_title[future]
                    try:
                        links = future.result()
                        incoming_map[title] = links
                    except Exception as exc:
                        if self.verbose: safe_print(f"Incoming fetch exception: {exc}")
                        incoming_map[title] = []

            # Process edges
            all_backward_edges = []
            unique_parents = set()
            
            for child, links_to_child in incoming_map.items():
                current_path_from_child = backward_frontier[child] 
                
                for parent_page in links_to_child:
                    if parent_page not in visited_backward:
                         # CHECK INTERSECTION with FULL FORWARD HISTORY
                        if parent_page in visited_forward:
                            path_from_start = visited_forward[parent_page]
                            # path_from_start ends with parent_page
                            # The backward path we are building is: parent_page -> child -> ... -> Target
                            # current_path_from_child is: child -> ... -> Target
                            
                            full_path = path_from_start + current_path_from_child
                            
                            elapsed = time.perf_counter() - start_time
                            if self.verbose:
                                safe_print(f"\n[SUCCESS] Meeting point found (Backward step): '{parent_page}'")
                                safe_print(f"Total time: {elapsed:.2f}s")
                            self.embedder.save_cache()
                            self.link_cache.save_cache()
                            
                            if return_history:
                                return full_path, "reached", history
                            return full_path, "reached"

                        all_backward_edges.append((parent_page, child))
                        unique_parents.add(parent_page)
                        if return_history:
                            current_history_step_back["candidates"].append((parent_page, child)) # parent -> child

            if not unique_parents:
                if self.verbose: safe_print("  [<] Backward search dead end.")
                backward_frontier = {}
            else:
                # Score parents against START
                parent_list = list(unique_parents)
                embeddings = self.embedder.get_or_compute_batch([(t, t) for t in parent_list])
                parent_vecs = np.array([embeddings[t] for t in parent_list])
                scores = cosine_similarity_vectorized(parent_vecs, start_vec)
                parent_scores = {title: scores[i] for i, title in enumerate(parent_list)}
                
                # Keep top K
                best_parents = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)[:beam_width]
                best_parents_set = set(t for t, _ in best_parents)
                
                if return_history:
                    current_history_step_back["selected"] = list(best_parents_set)
                
                for parent, child in all_backward_edges:
                    if parent in best_parents_set and parent not in next_backward_frontier:
                        new_path = [parent] + backward_frontier[child]
                        next_backward_frontier[parent] = new_path
                        visited_backward[parent] = new_path # Save to history
                
                backward_frontier = next_backward_frontier
                
                if self.verbose:
                    best_score = best_parents[0][1] if best_parents else 0
                    safe_print(f"  [<] Backward best score: {best_score:.4f} ('{best_parents[0][0]}' if any)")

            if return_history:
                history.append(current_history_step_back)
            
            if not forward_frontier and not backward_frontier:
                 if self.verbose: safe_print("Both directions stuck.")
                 break

        self.embedder.save_cache()
        self.link_cache.save_cache()
        if return_history:
            return [], "max_steps", history
        return [], "max_steps"
