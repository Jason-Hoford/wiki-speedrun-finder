import argparse
import time
from bidirectional_walker import BidirectionalWalker
from safe_print import safe_print
import wiki_api

def main():
    parser = argparse.ArgumentParser(description="Wiki Speed-Run Finder (Bidirectional)")
    parser.add_argument("start", help="Start page title")
    parser.add_argument("target", help="Target page title")
    parser.add_argument("--max_steps", type=int, default=10, help="Max depth from EACH side")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width")
    args = parser.parse_args()

    start_title = args.start
    target_title = args.target

    safe_print(f"Goal: {start_title} -> {target_title}")
    safe_print("Using Bidirectional Semantic Beam Search")
    
    # Initialize walker
    walker = BidirectionalWalker()
    
    start_time = time.time()
    path, status = walker.walk(
        start_title=start_title,
        target_title=target_title,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        verbose=True
    )
    end_time = time.time()
    
    print("\n" + "="*60)
    print(f"FINAL RESULT: {status.upper()}")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print("="*60)
    
    if path:
        print(f"Path Length: {len(path)-1} steps")
        print("Path:")
        for i, title in enumerate(path):
            safe_print(f"  {i}. {title}")
            if i < len(path) - 1:
                print("      |")
                print("      v")
        
        # Verify path validity (optional)
        print("\nVerifying path links...")
        valid = True
        for i in range(len(path) - 1):
            curr = path[i]
            next_p = path[i+1]
            links = wiki_api.get_outgoing_links(curr, use_cache=True)
            if next_p not in links:
                print(f"[ERROR] Link broken: '{curr}' does NOT link to '{next_p}'")
                valid = False
        
        if valid:
            print("Path Verified: VALID")
        else:
            print("Path Verified: INVALID (Possible hallucination or cache issue)")
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
