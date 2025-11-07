from pathlib import Path
from typing import Optional, List, Union
import numpy as np
from loguru import logger

from .hailo_inference import HailoInference
from .qdrant_image_store import QdrantImageStore


class VisualSearchEngine:
    """
    High-level engine for image-based visual search.

    Combines:
    - HailoInference → for computing embeddings
    - QdrantImageStore → for storing & searching embeddings

    Provides:
    - add_file()
    - add_dir()
    - get_embedding()
    - search()
    - delete_file()
    """

    def __init__(
        self,
        hef_path: Path,
        base_folder: Path,
        collection_name: str = "images",
        qdrant_url: str = "http://localhost:6333",
        vector_size: int = 512,
        distance_metric: str = "COSINE",
        image_size=(224, 224),
        profile_batch_size: int = 100,
        max_images: Optional[int] = 5000,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        max_segment_size: int = 100000,
    ):
        """Initialize both Qdrant store and Hailo inference modules."""
        self.base_folder = Path(base_folder).resolve()

        # --- Initialize Qdrant store ---
        from qdrant_client.models import Distance

        distance_enum = getattr(Distance, distance_metric.upper(), Distance.COSINE)
        self.store = QdrantImageStore(
            collection_name=collection_name,
            url=qdrant_url,
            vector_size=vector_size,
            distance=distance_enum,
            hnsw_m=hnsw_m,
            hnsw_ef_construct=hnsw_ef_construct,
            max_segment_size=max_segment_size,
        )

        # --- Initialize Hailo inference ---
        self.inference = HailoInference(
            hef_path=hef_path,
            base_folder=self.base_folder,
            image_size=image_size,
            profile_batch_size=profile_batch_size,
            max_images=max_images,
        )

    # ---------------------------------------------------------------------
    def add_file(self, rel_path: Path):
        """Compute and store embedding for a single image file."""
        path = self.base_folder / rel_path
        if not path.is_file():
            logger.warning(f"add_file: {rel_path} is not a valid file — skipping.")
            return

        logger.info(f"Adding file: {rel_path}")

        def _callback(path, vec_f32):
            self.store.add_vector(path, vec_f32, base=self.base_folder)

        self.inference.process(rel_path, callback=_callback)

    # ---------------------------------------------------------------------
    def add_dir(self, rel_dir: Path):
        """Compute and store embeddings for all images in a directory."""
        path = self.base_folder / rel_dir
        if not path.is_dir():
            logger.warning(f"add_dir: {rel_dir} is not a valid directory — skipping.")
            return

        logger.info(f"Adding directory: {rel_dir}")

        def _callback(path, vec_f32):
            self.store.add_vector(path, vec_f32, base=self.base_folder)

        self.inference.process(rel_dir, callback=_callback)

    # ---------------------------------------------------------------------
    def get_embedding(
        self, path: Path, store_if_missing: bool = True
    ) -> Optional[np.ndarray]:
        """
        Retrieve embedding for an image.

        - If path is inside base_folder, try fetching from Qdrant.
          If not found, compute embedding (store only if store_if_missing=True).
        - If path is outside base_folder, compute embedding only (never store).
        """
        path = Path(path).resolve()
        try:
            rel_path = path.relative_to(self.base_folder)
            is_within_base = True
        except ValueError:
            is_within_base = False

        if is_within_base:
            # Try to retrieve from Qdrant
            point = self.store.get_vector(rel_path, base=self.base_folder)
            if point and hasattr(point[0], "vector") and point[0].vector is not None:
                logger.debug(f"Embedding already exists for {rel_path}")
                return np.array(point[0].vector, dtype=np.float32)

            logger.info(f"Embedding missing for {rel_path}, generating now...")

            # Compute via inference
            result_container = {}

            def _capture_embedding(path, vec_f32):
                result_container["embedding"] = vec_f32
                if store_if_missing:
                    self.store.add_vector(path, vec_f32, base=self.base_folder)

            self.inference.process(rel_path, callback=_capture_embedding)
            return result_container.get("embedding")

        else:
            # External image — compute only
            logger.info(f"Computing embedding for external image: {path}")
            result_container = {}

            def _capture_embedding(path, vec_f32):
                result_container["embedding"] = vec_f32

            self.inference.process(path, callback=_capture_embedding)
            return result_container.get("embedding")

    # ---------------------------------------------------------------------
    def search(
        self,
        query: Union[Path, np.ndarray],
        limit: int = 5,
    ) -> List[dict]:
        """
        Search similar images in the vector store.

        - If `query` is a Path: compute embedding only (never store).
        - If `query` is an np.ndarray: search directly.
        """
        if isinstance(query, Path):
            logger.info(f"Preparing query embedding for {query}")
            query_vec = self.get_embedding(query, store_if_missing=False)
            if query_vec is None:
                logger.warning(f"Failed to compute embedding for {query}")
                return []
        else:
            query_vec = query

        results = self.store.search(query_vec, limit=limit)
        logger.info(f"Found {len(results)} similar images for query.")
        return results

    # ---------------------------------------------------------------------
    def delete_file(self, rel_path: Path):
        """Delete a file’s embedding from Qdrant if the file exists."""
        path = self.base_folder / rel_path
        if not path.is_file():
            logger.warning(f"delete_file: {rel_path} is not a valid file — skipping.")
            return

        logger.info(f"Deleting embedding for: {rel_path}")
        try:
            self.store.delete_vector(rel_path, base=self.base_folder)
            logger.info(f"Deleted embedding for: {rel_path}")
        except Exception as e:
            logger.error(f"Failed to delete {rel_path}: {e}")
