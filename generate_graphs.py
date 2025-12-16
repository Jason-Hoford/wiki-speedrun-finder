"""
Generate comprehensive statistical graphs from Wiki Speed-Run test results.
Creates individual graphs and a combined dashboard showing:
- Success rate heatmaps (9x9)
- Time taken heatmaps (9x9)
- Distribution histograms
- Performance metrics
- And more statistical visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.ndimage import gaussian_filter
from collections import Counter
import os

# Set style for better-looking graphs
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300  # High resolution
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 10)


def load_test_results(filename="test_results.json"):
    """Load test results from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] {filename} not found. Please run test_wiki_combinations.py first.")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {filename}: {e}")
        return None


def create_heatmap_data(results):
    """Create 3x3 matrices for success rate and average time."""
    categories = ["common", "semi_common", "random"]
    
    success_matrix = np.zeros((3, 3))
    time_matrix = np.zeros((3, 3))
    count_matrix = np.zeros((3, 3))
    
    # Map category names to indices
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    # Fill matrices
    for combo, agg in results["by_combination"].items():
        # Parse combination like "common -> semi_common"
        parts = combo.split(" -> ")
        if len(parts) != 2:
            continue
        
        src_cat, tgt_cat = parts[0].strip(), parts[1].strip()
        
        if src_cat in cat_to_idx and tgt_cat in cat_to_idx:
            i, j = cat_to_idx[src_cat], cat_to_idx[tgt_cat]
            
            total = agg["total_tests"]
            found = agg["found_count"]
            
            success_matrix[i, j] = (found / total * 100) if total > 0 else 0
            time_matrix[i, j] = agg["average_time"]
            count_matrix[i, j] = total
    
    return success_matrix, time_matrix, count_matrix, categories


def plot_success_rate_heatmap(success_matrix, categories, filename="success_rate_heatmap.png"):
    """Create a 9x9 success rate heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap (red to yellow to green)
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('success', colors, N=n_bins)
    
    # Create heatmap
    im = ax.imshow(success_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{success_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=12, weight='bold')
    
    # Labels and title
    ax.set_xlabel('Target Category', fontsize=12, weight='bold')
    ax.set_ylabel('Source Category', fontsize=12, weight='bold')
    ax.set_title('Success Rate Heatmap (%)\nSource → Target', fontsize=14, weight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Success Rate (%)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def plot_time_heatmap(time_matrix, categories, filename="time_taken_heatmap.png"):
    """Create a 9x9 average time heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap (green to yellow to red)
    colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('time', colors, N=n_bins)
    
    # Create heatmap
    im = ax.imshow(time_matrix, cmap=cmap, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{time_matrix[i, j]:.1f}s',
                          ha="center", va="center", color="black", fontsize=12, weight='bold')
    
    # Labels and title
    ax.set_xlabel('Target Category', fontsize=12, weight='bold')
    ax.set_ylabel('Source Category', fontsize=12, weight='bold')
    ax.set_title('Average Time Taken Heatmap (seconds)\nSource → Target', fontsize=14, weight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Time (seconds)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def plot_time_distribution(results, filename="time_distribution.png"):
    """Plot time distribution with bar chart and smooth line overlay."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get all times for successful solves
    times = [r["time_taken"] for r in results["all_results"] if r["found"]]
    
    if not times:
        print("[WARN] No successful solves to plot time distribution")
        return None
    
    # Create histogram bins
    bins = np.arange(0, max(times) + 20, 10)  # 10-second bins
    counts, bin_edges = np.histogram(times, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot bar chart
    bars = ax.bar(bin_centers, counts, width=8, alpha=0.6, color='steelblue', 
                   edgecolor='black', linewidth=0.5, label='Number of Solves')
    
    # Create smooth line using gaussian filter
    smooth_counts = gaussian_filter(counts.astype(float), sigma=1.5)
    ax.plot(bin_centers, smooth_counts, color='red', linewidth=2.5, 
            label='Trend Line', marker='o', markersize=4)
    
    # Formatting
    ax.set_xlabel('Time Taken (seconds)', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Solves', fontsize=12, weight='bold')
    ax.set_title('Distribution of Solve Times\nBar Chart with Trend Line', fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text
    mean_time = np.mean(times)
    median_time = np.median(times)
    stats_text = f'Mean: {mean_time:.1f}s\nMedian: {median_time:.1f}s\nTotal Solves: {len(times)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def plot_path_length_distribution(results, filename="path_length_distribution.png"):
    """Plot distribution of path lengths for successful solves."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get all path lengths for successful solves
    path_lengths = [r["path_length"] for r in results["all_results"] if r["found"]]
    
    if not path_lengths:
        print("[WARN] No successful solves to plot path length distribution")
        return None
    
    # Count frequency
    length_counts = Counter(path_lengths)
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    
    # Plot bar chart
    bars = ax.bar(lengths, counts, color='teal', alpha=0.7, edgecolor='black', linewidth=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Formatting
    ax.set_xlabel('Path Length (number of clicks)', fontsize=12, weight='bold')
    ax.set_ylabel('Number of Solves', fontsize=12, weight='bold')
    ax.set_title('Distribution of Path Lengths for Successful Solves', fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics
    mean_length = np.mean(path_lengths)
    median_length = np.median(path_lengths)
    mode_length = max(length_counts, key=length_counts.get)
    stats_text = f'Mean: {mean_length:.2f}\nMedian: {median_length:.1f}\nMode: {mode_length}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def plot_success_by_combination(results, filename="success_by_combination.png"):
    """Plot success rate for each combination."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get data
    combos = []
    success_rates = []
    
    for combo, agg in sorted(results["by_combination"].items()):
        combos.append(combo)
        total = agg["total_tests"]
        found = agg["found_count"]
        success_rate = (found / total * 100) if total > 0 else 0
        success_rates.append(success_rate)
    
    # Create bar chart with color gradient
    colors = plt.cm.RdYlGn(np.array(success_rates) / 100)
    bars = ax.bar(range(len(combos)), success_rates, color=colors, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        ax.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Formatting
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, rotation=45, ha='right')
    ax.set_ylabel('Success Rate (%)', fontsize=12, weight='bold')
    ax.set_title('Success Rate by Category Combination', fontsize=14, weight='bold', pad=15)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add horizontal line for average
    avg_success = np.mean(success_rates)
    ax.axhline(y=avg_success, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_success:.1f}%', alpha=0.7)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def plot_time_vs_success(results, filename="time_vs_success.png"):
    """Scatter plot of average time vs success rate for each combination."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get data
    success_rates = []
    avg_times = []
    labels = []
    sizes = []
    
    for combo, agg in results["by_combination"].items():
        total = agg["total_tests"]
        found = agg["found_count"]
        success_rate = (found / total * 100) if total > 0 else 0
        
        success_rates.append(success_rate)
        avg_times.append(agg["average_time"])
        labels.append(combo)
        sizes.append(total * 10)  # Size represents number of tests
    
    # Create scatter plot
    scatter = ax.scatter(avg_times, success_rates, s=sizes, alpha=0.6, 
                        c=success_rates, cmap='RdYlGn', edgecolors='black', linewidth=1)
    
    # Add labels for each point
    for i, label in enumerate(labels):
        ax.annotate(label, (avg_times[i], success_rates[i]), 
                   fontsize=8, alpha=0.7, ha='center')
    
    # Formatting
    ax.set_xlabel('Average Time (seconds)', fontsize=12, weight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, weight='bold')
    ax.set_title('Success Rate vs Average Time by Combination\n(Bubble size = number of tests)', 
                fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def plot_cumulative_time(results, filename="cumulative_time.png"):
    """Plot cumulative distribution of solve times."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get times for successful solves
    times = sorted([r["time_taken"] for r in results["all_results"] if r["found"]])
    
    if not times:
        print("[WARN] No successful solves to plot cumulative time")
        return None
    
    # Calculate cumulative percentages
    cumulative = np.arange(1, len(times) + 1) / len(times) * 100
    
    # Plot
    ax.plot(times, cumulative, linewidth=2.5, color='darkblue', marker='o', 
            markersize=3, markevery=max(1, len(times)//50))
    ax.fill_between(times, cumulative, alpha=0.3, color='lightblue')
    
    # Add reference lines
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        idx = int(len(times) * p / 100)
        if idx < len(times):
            time_at_p = times[idx]
            ax.axhline(y=p, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axvline(x=time_at_p, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.text(time_at_p + 5, p + 2, f'P{p}: {time_at_p:.0f}s', fontsize=8)
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12, weight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Cumulative Distribution of Solve Times', fontsize=14, weight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def plot_status_breakdown(results, filename="status_breakdown.png"):
    """Plot pie chart of solve statuses."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Count statuses
    status_counts = Counter()
    for r in results["all_results"]:
        if r["found"]:
            status_counts["Reached"] += 1
        elif r["status"] == "timeout":
            status_counts["Timeout"] += 1
        elif r["status"] == "stuck":
            status_counts["Stuck"] += 1
        elif r["status"] == "max_steps":
            status_counts["Max Steps"] += 1
        else:
            status_counts["Other"] += 1
    
    # Create pie chart
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
    explode = [0.05 if label == "Reached" else 0 for label in labels]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                       colors=colors, explode=explode, shadow=True,
                                       startangle=90, textprops={'weight': 'bold'})
    
    # Make percentage text larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
    
    ax.set_title('Test Results Status Breakdown', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved {filename}")
    return fig


def create_combined_dashboard(results, filename="combined_dashboard.png"):
    """Create a single image with all graphs combined."""
    # Create figure with complex grid
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get matrices
    success_matrix, time_matrix, count_matrix, categories = create_heatmap_data(results)
    
    # 1. Success Rate Heatmap (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    cmap1 = LinearSegmentedColormap.from_list('success', 
                                              ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'], N=100)
    im1 = ax1.imshow(success_matrix, cmap=cmap1, aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_yticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories, fontsize=8)
    ax1.set_yticklabels(categories, fontsize=8)
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax1.text(j, i, f'{success_matrix[i, j]:.0f}%', ha="center", va="center", fontsize=9, weight='bold')
    ax1.set_title('Success Rate Heatmap (%)', fontsize=11, weight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Time Heatmap (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    cmap2 = LinearSegmentedColormap.from_list('time',
                                              ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027'], N=100)
    im2 = ax2.imshow(time_matrix, cmap=cmap2, aspect='auto')
    ax2.set_xticks(np.arange(len(categories)))
    ax2.set_yticks(np.arange(len(categories)))
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_yticklabels(categories, fontsize=8)
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax2.text(j, i, f'{time_matrix[i, j]:.0f}s', ha="center", va="center", fontsize=9, weight='bold')
    ax2.set_title('Average Time Heatmap (seconds)', fontsize=11, weight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Time Distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    times = [r["time_taken"] for r in results["all_results"] if r["found"]]
    if times:
        bins = np.arange(0, max(times) + 20, 10)
        counts, bin_edges = np.histogram(times, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax3.bar(bin_centers, counts, width=8, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5)
        smooth_counts = gaussian_filter(counts.astype(float), sigma=1.5)
        ax3.plot(bin_centers, smooth_counts, color='red', linewidth=2, marker='o', markersize=3)
        ax3.set_xlabel('Time (seconds)', fontsize=9)
        ax3.set_ylabel('Number of Solves', fontsize=9)
        ax3.set_title('Time Distribution with Trend', fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. Path Length Distribution (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    path_lengths = [r["path_length"] for r in results["all_results"] if r["found"]]
    if path_lengths:
        length_counts = Counter(path_lengths)
        lengths = sorted(length_counts.keys())
        counts = [length_counts[l] for l in lengths]
        ax4.bar(lengths, counts, color='teal', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Path Length (clicks)', fontsize=9)
        ax4.set_ylabel('Number of Solves', fontsize=9)
        ax4.set_title('Path Length Distribution', fontsize=11, weight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Success by Combination (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    combos = []
    success_rates = []
    for combo, agg in sorted(results["by_combination"].items()):
        combos.append(combo)
        total = agg["total_tests"]
        found = agg["found_count"]
        success_rates.append((found / total * 100) if total > 0 else 0)
    colors = plt.cm.RdYlGn(np.array(success_rates) / 100)
    ax5.bar(range(len(combos)), success_rates, color=colors, edgecolor='black')
    ax5.set_xticks(range(len(combos)))
    ax5.set_xticklabels(combos, rotation=45, ha='right', fontsize=7)
    ax5.set_ylabel('Success Rate (%)', fontsize=9)
    ax5.set_title('Success Rate by Combination', fontsize=11, weight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Time vs Success Scatter (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    success_rates_scatter = []
    avg_times_scatter = []
    sizes_scatter = []
    for combo, agg in results["by_combination"].items():
        total = agg["total_tests"]
        found = agg["found_count"]
        success_rates_scatter.append((found / total * 100) if total > 0 else 0)
        avg_times_scatter.append(agg["average_time"])
        sizes_scatter.append(total * 10)
    scatter = ax6.scatter(avg_times_scatter, success_rates_scatter, s=sizes_scatter, 
                         alpha=0.6, c=success_rates_scatter, cmap='RdYlGn', edgecolors='black')
    ax6.set_xlabel('Avg Time (seconds)', fontsize=9)
    ax6.set_ylabel('Success Rate (%)', fontsize=9)
    ax6.set_title('Time vs Success', fontsize=11, weight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative Time Distribution (second bottom left)
    ax7 = fig.add_subplot(gs[3, 0])
    if times:
        sorted_times = sorted(times)
        cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times) * 100
        ax7.plot(sorted_times, cumulative, linewidth=2, color='darkblue')
        ax7.fill_between(sorted_times, cumulative, alpha=0.3, color='lightblue')
        ax7.set_xlabel('Time (seconds)', fontsize=9)
        ax7.set_ylabel('Cumulative %', fontsize=9)
        ax7.set_title('Cumulative Time Distribution', fontsize=11, weight='bold')
        ax7.grid(True, alpha=0.3)
    
    # 8. Status Breakdown Pie (second bottom right)
    ax8 = fig.add_subplot(gs[3, 1])
    status_counts = Counter()
    for r in results["all_results"]:
        if r["found"]:
            status_counts["Reached"] += 1
        elif r["status"] == "timeout":
            status_counts["Timeout"] += 1
        elif r["status"] == "stuck":
            status_counts["Stuck"] += 1
        else:
            status_counts["Other"] += 1
    labels = list(status_counts.keys())
    sizes = list(status_counts.values())
    colors_pie = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    ax8.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax8.set_title('Status Breakdown', fontsize=11, weight='bold')
    
    # Overall title
    fig.suptitle('Wiki Speed-Run Comprehensive Analysis Dashboard', 
                fontsize=16, weight='bold', y=0.995)
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] Saved combined dashboard: {filename}")
    return fig


def main():
    """Generate all graphs from test results."""
    print("=" * 80)
    print("=== Wiki Speed-Run Test Results Graph Generator ===")
    print("=" * 80)
    
    # Load results
    print("\n[INFO] Loading test results...")
    results = load_test_results()
    
    if not results:
        print("[ERROR] Could not load test results. Exiting.")
        return
    
    print(f"[INFO] Loaded {results['metadata']['total_tests']} test results")
    print(f"[INFO] Success rate: {results['summary']['overall_success_rate']:.1f}%")
    
    # Create output directory for individual graphs
    output_dir = "graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created directory: {output_dir}/")
    
    # Get heatmap data
    success_matrix, time_matrix, count_matrix, categories = create_heatmap_data(results)
    
    print("\n[INFO] Generating individual graphs...")
    
    # Generate all individual graphs
    plot_success_rate_heatmap(success_matrix, categories, f"{output_dir}/1_success_rate_heatmap.png")
    plot_time_heatmap(time_matrix, categories, f"{output_dir}/2_time_taken_heatmap.png")
    plot_time_distribution(results, f"{output_dir}/3_time_distribution.png")
    plot_path_length_distribution(results, f"{output_dir}/4_path_length_distribution.png")
    # plot_success_by_combination(results, f"{output_dir}/5_success_by_combination.png")
    plot_time_vs_success(results, f"{output_dir}/6_time_vs_success_scatter.png")
    plot_cumulative_time(results, f"{output_dir}/7_cumulative_time_distribution.png")
    # plot_status_breakdown(results, f"{output_dir}/8_status_breakdown_pie.png")
    
    print("\n[INFO] Generating combined dashboard...")
    create_combined_dashboard(results, "combined_dashboard.png")
    
    print("\n" + "=" * 80)
    print("=== Graph Generation Complete ===")
    print("=" * 80)
    print(f"Individual graphs saved to: {output_dir}/")
    print(f"Combined dashboard saved to: combined_dashboard.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
