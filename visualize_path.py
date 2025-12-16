import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from bidirectional_walker import BidirectionalWalker
from safe_print import safe_print
import wiki_api

def visualize_search(start_title: str, target_title: str, output_file: str = "path_visualization.png"):
    # 1. Run Search with History
    print("Running Bidirectional Search...")
    walker = BidirectionalWalker()
    path, status, history = walker.walk(
        start_title=start_title,
        target_title=target_title,
        max_steps=10,
        beam_width=10,
        verbose=True,
        return_history=True
    )
    
    if not history:
        print("No history captured (search failed immediately or finished without steps).")
        return

    print(f"Search status: {status}")
    print(f"History steps: {len(history)}")

    # 2. Collect all unique nodes to vectorize
    all_nodes = set()
    node_metadata = {} # title -> {'type': 'candidate'|'selected'|'start'|'target'|'path', 'step': int}
    
    # Edges for plotting lines
    candidate_edges = [] # (parent, child)
    selected_edges = [] # (parent, child)
    
    # Add Start/Target
    start_canonical = path[0] if path else start_title
    target_canonical = path[-1] if path else target_title
    
    all_nodes.add(start_canonical)
    all_nodes.add(target_canonical)
    node_metadata[start_canonical] = {'type': 'start', 'step': 0}
    node_metadata[target_canonical] = {'type': 'target', 'step': 0}

    # Process history
    for step_data in history:
        step_num = step_data['step']
        direction = step_data['direction']
        selected_set = set(step_data['selected'])
        
        for child, parent in step_data['candidates']:
            all_nodes.add(child)
            all_nodes.add(parent)
            
            # Record edges
            if child in selected_set:
                selected_edges.append((parent, child))
                node_type = 'selected'
            else:
                candidate_edges.append((parent, child))
                node_type = 'candidate'
            
            # Update metadata if not set or if upgrading importance
            if child not in node_metadata or node_metadata[child]['type'] == 'candidate':
                node_metadata[child] = {'type': node_type, 'step': step_num}

    # Mark path nodes
    path_edges = []
    if path:
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            path_edges.append((u, v))
            node_metadata[u] = {'type': 'path', 'step': -1}
            node_metadata[v] = {'type': 'path', 'step': -1}
            all_nodes.add(u)
            all_nodes.add(v)

    # 3. Compute Embeddings
    print(f"Computing/Retrieving embeddings for {len(all_nodes)} nodes...")
    node_list = list(all_nodes)
    embeddings_map = walker.embedder.get_or_compute_batch([(t, t) for t in node_list])
    
    X = np.array([embeddings_map[t] for t in node_list])
    
    # 4. PCA Projection
    print("Projecting to 2D...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Map title -> (x, y)
    coords = {title: X_2d[i] for i, title in enumerate(node_list)}
    
    # 5. Plotting
    print("Generating plot...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Helper to plot edges
    def plot_edges(edges, color, alpha, linewidth, zorder):
        for u, v in edges:
            if u in coords and v in coords:
                p1, p2 = coords[u], coords[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color, alpha=alpha, linewidth=linewidth, zorder=zorder)

    # Plot candidate edges (faint connection lines)
    # Using a subset if too many?
    if len(candidate_edges) > 2000:
        import random
        plot_candidates = random.sample(candidate_edges, 2000)
    else:
        plot_candidates = candidate_edges
    
    plot_edges(plot_candidates, color='#33ff33', alpha=0.08, linewidth=0.5, zorder=1) # Very faint green

    # Plot selected edges (stronger lines)
    plot_edges(selected_edges, color='#33ff33', alpha=0.3, linewidth=1.0, zorder=2) # Visible green

    # Plot Path Edges (Bright white/thick)
    # Arrow style for path?
    for u, v in path_edges:
        if u in coords and v in coords:
            p1, p2 = coords[u], coords[v]
            ax.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1], 
                     color='white', alpha=0.9, width=0.002, head_width=0.015, zorder=10,
                     length_includes_head=True)

    # Plot Nodes
    # Candidates
    cand_x, cand_y = [], []
    sel_x, sel_y = [], []
    
    for title, meta in node_metadata.items():
        if title in coords:
            x, y = coords[title]
            if meta['type'] == 'candidate':
                cand_x.append(x)
                cand_y.append(y)
            elif meta['type'] == 'selected':
                sel_x.append(x)
                sel_y.append(y)

    ax.scatter(cand_x, cand_y, c='#33ff33', s=5, alpha=0.2, zorder=3, label='Candidates')
    ax.scatter(sel_x, sel_y, c='#33ff33', s=20, alpha=0.6, zorder=4, label='Selected')

    # Start/Target/Path Nodes
    path_x, path_y = [], []
    path_labels = []
    
    start_pos = coords.get(start_canonical)
    target_pos = coords.get(target_canonical)
    
    if start_pos is not None:
        ax.scatter([start_pos[0]], [start_pos[1]], c='gold', s=150, marker='*', zorder=20, label='Start')
        ax.text(start_pos[0], start_pos[1], f"  {start_canonical}", color='gold', fontsize=10, fontweight='bold', zorder=21)
        
    if target_pos is not None:
        ax.scatter([target_pos[0]], [target_pos[1]], c='red', s=150, marker='*', zorder=20, label='Target')
        ax.text(target_pos[0], target_pos[1], f"  {target_canonical}", color='red', fontsize=10, fontweight='bold', zorder=21)

    # Label path nodes
    if path:
        for title in path:
            if title in coords and title != start_canonical and title != target_canonical:
                p = coords[title]
                ax.scatter([p[0]], [p[1]], c='white', s=50, zorder=15)
                ax.text(p[0], p[1], f"  {title}", color='white', fontsize=9, zorder=16)

    ax.set_title(f"Bidirectional Search Visualization: {start_title} -> {target_title}", color='white', fontsize=16)
    ax.axis('off') # Remove axes for cleaner look
    
    # Legend
    leg = ax.legend(facecolor='black', edgecolor='white')
    for text in leg.get_texts():
        text.set_color("white")
        
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, facecolor='black')
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="Start page")
    parser.add_argument("target", help="Target page")
    parser.add_argument("--out", default="path_visualization.png", help="Output file")
    args = parser.parse_args()
    
    visualize_search(args.start, args.target, args.out)
