import time
from pathlib import Path
from visual_search_engine import SimilaritySearchEngine

base_folder = Path("/home/anandas/test_images")

engine = SimilaritySearchEngine(
    base_folder=base_folder,
    qdrant_url="http://localhost:6333",
)
start = time.perf_counter()

engine.add_file(base_folder / "Datewise/12/27/2016/IMG_0582.jpg")

end = time.perf_counter()
print(f"✅ Completed indexing in {end - start:.2f} seconds")
