import sys
import time
import argparse
from pathlib import Path

from implementation.similarity_search import SimilaritySearch


def main():
    parser = argparse.ArgumentParser(description="Visual Search Engine Demo")
    parser.add_argument(
        "--ss", action="store_true", help="Enable similarity search functionality."
    )
    parser.add_argument(
        "--add",
        nargs="+",
        help="Add file(s) or directory(ies) to the search engine. Requires --ss.",
    )
    parser.add_argument("--search", help="Search for a similar image. Requires --ss.")
    parser.add_argument(
        "--rootdir", required=True, help="Base directory for the search engine"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace the existing embedding"
    )

    args = parser.parse_args()

    if (args.add or args.search) and not args.ss:
        print("Error: --add and --search options require --ss to be specified.")
        return

    if not args.ss:
        print("Please specify --ss for similarity search")
        return

    if not args.add and args.force:
        print("--force is used with --add. Ignoring")

    base_folder = Path(args.rootdir)

    engine = SimilaritySearch(base_folder=base_folder)

    if args.add:
        for path_str in args.add:
            path = Path(path_str)
            if path.is_dir():
                engine.add_dir(path, force=args.force)
            elif path.is_file():
                engine.add_file(path, force=args.force)

    if args.search:
        results = engine.search(Path(args.search))
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
