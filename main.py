import time
import argparse
from pathlib import Path

from visual_search_engine_file_system import VisualSearchEngineFileSystem
from implementation.inference.hailo_inference import HailoInference
from implementation.store.qdrant_image_store import QdrantImageStore
from implementation.progress_bar.progress_bar import ProgressBar
from qdrant_client.models import Distance
from logger import logger
from config import Config


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

    args = parser.parse_args()

    if (args.add or args.search) and not args.ss:
        print("Error: --add and --search options require --ss to be specified.")
        return

    if not args.ss:
        print("Please specify --ss for similarity search")
        return

    base_folder = Path(args.rootdir)

    # Initialize HailoInference
    hailo_inference_engine = HailoInference(
        hef_path=Config.DEFAULT_HEF_PATH,
        profile_batch_size=100,
        max_items=None,
        logger=logger,
    )

    # Initialize QdrantImageStore
    qdrant_store = QdrantImageStore(
        collection_name="images2",
        url="http://localhost:6333",
        vector_size=512,  # Default vector size for the model
        distance=Distance.COSINE,
        logger=logger,
    )

    engine = VisualSearchEngineFileSystem(
        inference_engine=hailo_inference_engine,
        store_interface=qdrant_store,
        base_folder=base_folder,
        logger=logger,
        progress_bar_class=ProgressBar,
    )

    if args.add:
        start = time.perf_counter()
        for path_str in args.add:
            path = Path(path_str)
            if path.is_dir():
                engine.add_dir(path)
            elif path.is_file():
                engine.add_file(path)
        end = time.perf_counter()
        print(f"✅ Completed indexing in {end - start:.2f} seconds")

    if args.search:
        start = time.perf_counter()
        results = engine.search(Path(args.search))
        end = time.perf_counter()
        print(f"✅ Search completed in {end - start:.2f} seconds")
        for result in results:
            print(result)


if __name__ == "__main__":
    main()
