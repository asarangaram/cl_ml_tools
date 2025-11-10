from pathlib import Path
from typing import Optional, List, Union
import numpy as np
from .logger import logger
import hashlib

from .hailo_inference import HailoInference
from .qdrant_image_store import QdrantImageStore


class VisualSearchEngine:
    """
    Base class for visual search engines.

    Model-agnostic: does not assume a specific HEF file or vector type.
    Subclasses should provide appropriate model, collection_name, and dimensions.
    """

    def __init__(
        self,
        hef_path: Path,
        base_folder: Path,
        collection_name: str,
        vector_size: int,
        qdrant_url: str = "http://localhost:6333",
        distance_metric: str = "COSINE",
        profile_batch_size: int = 100,
        max_images: Optional[int] = None,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        max_segment_size: int = 100000,
    ):
        """Initialize both inference and vector store."""
        self.base_folder = Path(base_folder).resolve()
        self.hef_path = Path(hef_path).resolve()
        self.collection_name = collection_name

        # --- Initialize Qdrant Store ---
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

        # --- Initialize Hailo Inference ---
        self.inference = HailoInference(
            hef_path=self.hef_path,
            profile_batch_size=profile_batch_size,
            max_images=max_images,
        )

    # ---------------------------------------------------------------------
    @staticmethod
    def make_id(path: Union[str, Path]) -> int:
        """Compute a deterministic ID from a file path."""
        normalized = str(path).strip().replace("\\", "/").lower()
        return int(hashlib.sha1(normalized.encode()).hexdigest(), 16) % (2**63)

    # ---------------------------------------------------------------------
    def _relative_path(self, path: Path) -> Path:
        """Compute path relative to base folder, or absolute if outside."""
        try:
            return path.resolve().relative_to(self.base_folder)
        except ValueError:
            return path.resolve()

    # ---------------------------------------------------------------------
    def add_file(self, image_path: Path):
        """Compute and store a single file embedding."""
        if not image_path.is_file():
            logger.warning(f"add_file: {image_path} is not a valid file.")
            return

        vec_f32 = self.inference.process_file(image_path)
        if vec_f32 is None:
            logger.warning(f"Failed to generate embedding for {image_path}")
            return

        rel_path = self._relative_path(image_path)
        point_id = self.make_id(rel_path)
        self.store.add_vector(point_id, vec_f32, payload={"filename": str(rel_path)})

        logger.debug(f"Added embedding for {rel_path}")

    # ---------------------------------------------------------------------
    def add_dir(self, dir_path: Path):
        """Compute and store embeddings for all images in a directory."""
        if not dir_path.is_dir():
            logger.warning(f"add_dir: {dir_path} is not a valid directory.")
            return

        def _callback(image_path, vec_f32):
            rel_path = self._relative_path(image_path)
            point_id = self.make_id(rel_path)
            self.store.add_vector(
                point_id, vec_f32, payload={"filename": str(rel_path)}
            )

        logger.debug(f"Indexing directory: {dir_path}")
        self.inference.process_dir(dir_path, callback=_callback)

    # ---------------------------------------------------------------------
    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """Fetch from store if possible; else compute embedding on the fly."""
        rel_path = self._relative_path(image_path)
        is_within_base = rel_path != image_path

        if is_within_base:
            point_id = self.make_id(rel_path)
            point = self.store.get_vector(point_id)
            if point and hasattr(point[0], "vector") and point[0].vector is not None:
                return np.array(point[0].vector, dtype=np.float32)
            logger.debug(f"Embedding not found in store for {rel_path}, computing...")

        return self.inference.process_file(image_path)

    # ---------------------------------------------------------------------
    def search(self, query: Union[Path, np.ndarray], limit: int = 5) -> List[dict]:
        """Search similar vectors (does not modify store)."""
        if isinstance(query, Path):
            logger.debug(f"Preparing query embedding for {query}")
            query_vec = self.inference.process_file(query)
            if query_vec is None:
                logger.warning(f"Failed to compute embedding for {query}")
                return []
        else:
            query_vec = query

        search_results = self.store.search(query_vec, limit=limit)

        results = [
            {**r, "filename": self.base_folder / r["filename"]}
            for r in search_results
            if r.get("filename")
        ]
        if isinstance(query, Path):
            results = [
                result
                for result in results
                if result["filename"].resolve() != query.resolve()
            ]

        logger.debug(f"Found {len(results)} results for query.")
        return results[:limit]

    # ---------------------------------------------------------------------
    def delete_file(self, image_path: Path):
        """Delete a stored embedding by file path."""
        if not image_path.is_file():
            logger.warning(f"delete_file: {image_path} is not a valid file.")
            return

        rel_path = self._relative_path(image_path)
        point_id = self.make_id(rel_path)
        self.store.delete_vector(point_id)
        logger.debug(f"Deleted embedding for {rel_path}")
