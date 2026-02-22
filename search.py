"""
Clipper — Search Script

Usage:
    python search.py                           interactive mode
    python search.py "a dog running"           single query
    python search.py "a dog running" --k 5     return top 5
    python search.py "sunset" --no-rerank      skip local reranking

Each result tells you exactly which video file to open and
at what timestamp the matching clip starts and ends.
"""
import argparse

from config    import ClipperConfig
from retriever import ClipperRetriever


def fmt_time(s: float) -> str:
    """Converts seconds to MM:SS string."""
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"


def print_results(results: list, query: str = ""):
    if not results:
        print("\nNo results found.")
        return

    print(f"\n{'─'*52}")
    if query:
        print(f"  Query  : \"{query}\"")
        print(f"{'─'*52}")
    print(f"  Found {len(results)} clip(s)\n")

    for i, r in enumerate(results, 1):
        duration = round(r["t_end"] - r["t_start"], 1)
        print(f"  Rank {i}  |  Score : {r['score']:.4f}")
        print(f"  Video  : {r['video_path']}")
        print(f"  Clip   : {fmt_time(r['t_start'])} → "
              f"{fmt_time(r['t_end'])}  ({duration}s)")
        if r.get("local_score") is not None:
            print(f"  Global : {r['global_score']:.4f}  "
                  f"Local  : {r['local_score']:.4f}")
        print(f"{'─'*52}")


def interactive(retriever: ClipperRetriever, top_k: int, rerank: bool):
    print("\nClipper — Text to Video Clip Retrieval")
    print("Type a description and press Enter.")
    print("Commands: 'quit' to exit, 'help' for tips\n")

    tips = [
        "Be descriptive: 'a person in red jacket running on beach'",
        "Include actions: 'crowd cheering after goal'",
        "Include scenes: 'traffic at night in city',",
        "Use --no-rerank for faster (slightly less accurate) results"
    ]

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if query.lower() == "help":
            print("\nTips:")
            for t in tips:
                print(f"  • {t}")
            print()
            continue

        results = retriever.search(query, top_k=top_k, rerank=rerank)
        print_results(results, query)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Clipper — Text to Video Clip Retrieval",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Text query (omit for interactive mode)"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Skip local MaxSim reranking (faster)"
    )
    args = parser.parse_args()

    config    = ClipperConfig()
    retriever = ClipperRetriever(config)

    rerank = not args.no_rerank

    if args.query:
        results = retriever.search(
            args.query,
            top_k  = args.k,
            rerank = rerank
        )
        print_results(results, args.query)
    else:
        interactive(retriever, args.k, rerank)


if __name__ == "__main__":
    main()
