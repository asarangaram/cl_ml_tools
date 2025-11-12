from pathlib import Path
from typing import Optional, List, Union, Dict
import numpy as np

from .visual_search_engine import VisualSearchEngine

from .progress_bar import ProgressBar
from ..logger import logger
import hashlib
import time
from PIL import Image
import io
from qdrant_client.models import Distance

from .ml_inference import MLInference
from .store_interface import StoreInterface


class VisualSearchEngineFileSystem:
    """
    Base class for visual search engines.

    Model-agnostic: does not assume a specific HEF file or vector type.
    Subclasses should provide appropriate model, collection_name, and dimensions.
    """

    def __init__(
        self,
        inference_engine: MLInference,
        store_interface: StoreInterface,
        base_folder: Path,
    ):
        """Initialize both inference and vector store."""
        self.base_folder = Path(base_folder).resolve()
        self.engine = VisualSearchEngine(
            inference_engine=inference_engine, store_interface=store_interface
        )

    # ---------------------------------------------------------------------
    @staticmethod
    def make_id(path: Union[str, Path]) -> int:
        """Compute a deterministic ID from a file path."""
        normalized = str(path).strip().replace("\\", "/").lower()
        return int(hashlib.sha1(normalized.encode()).hexdigest(), 16) % (2**63)

    # ---------------------------------------------------------------------
    def _relative_path(self, path: Path) -> Optional[Path]:
        """Compute path relative to base folder, or absolute if outside."""
        try:
            return path.resolve().relative_to(self.base_folder)
        except ValueError:
            return None

    def add_file(self, image_path: Path, force: bool = False) -> bool:
        """
        Computes and stores the embedding for a single image file.

        By default, this method will skip processing if an embedding for the
        given image path already exists in the store. The image's path
        relative to `base_folder` is used to generate a deterministic ID.

        Args:
            image_path: The absolute path to the image file.
            force: If True, re-computes and updates the embedding even if it
                   already exists. Defaults to False.
        """
        if not image_path.is_file():
            logger.warning(f"{image_path} is not a valid file.")
            return

        rel_path = self._relative_path(image_path)
        if rel_path:
            point_id = self.make_id(rel_path)
            return self.engine.add_file(id=point_id, path=image_path, force=force)
        else:
            return False

    def _process_batch(self, buffers: List[np.ndarray], paths: List[Path]) -> int:
        """Process a batch of images and store their embeddings."""
        image_buffers_dict = {str(path): buffer for path, buffer in zip(paths, buffers)}
        batch_results = self.inference.infer_batch(image_buffers_dict)
        successful_embeddings = 0
        for image_path, vec_f32 in batch_results.items():
            if vec_f32 is not None:
                rel_path = self._relative_path(image_path)
                if rel_path:
                    point_id = self.make_id(rel_path)
                    self.store.add_vector(
                        point_id, vec_f32, payload={"filename": str(rel_path)}
                    )
                    successful_embeddings += 1
        return successful_embeddings

    def add_dir(self, dir_path: Path, force: bool = False, batch_size: int = 32):
        if not dir_path.is_dir():
            logger.warning(f"{dir_path} is not a valid directory.")
            return

        if not self._relative_path(dir_path):
            logger.warning(f"{dir_path} is not a managed directory")
            return

        logger.info(f"Starting to index directory: {dir_path}")

        files_to_process = [
            p for ext in ("*.jpg", "*.jpeg", "*.png") for p in dir_path.rglob(ext)
        ]

        if not files_to_process:
            logger.info("No new images to process.")
            return

        files = {}
        payloads = {}
        for file in files_to_process:
            rel_path = self._relative_path(file)
            if rel_path:
                id = self.make_id(rel_path)
                files[id] = file
                payloads[id] = {"filename": str(rel_path)}

        self.engine.add_all(files=files, payload=payloads)

    # ---------------------------------------------------------------------
    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        self.engine.get_embedding(image_path=image_path)

    # ---------------------------------------------------------------------
    def search(self, query: Union[Path, np.ndarray], limit: int = 5) -> List[dict]:
        """
        Finds similar images in the vector store.

        The query can be either a path to an image file or a pre-computed
        numpy array embedding. If a path is provided, the embedding is
        computed first. The search results are returned as a list of
        dictionaries, each containing the ID, score, and filename of a
        matching image.

        Args:
            query: The image to search for, as a file path or numpy array.
            limit: The maximum number of search results to return.

        Returns:
            A list of dictionaries, where each dictionary represents a
            similar image and contains its ID, score, and absolute filename.
        """
        query_id = None
        if isinstance(query, Path):
            logger.debug(f"Preparing query embedding for {query}")
            rel_path = self._relative_path(query)
            if rel_path:
                query_id = self.make_id(rel_path)

        search_results = self.store.search(
            query, limit=limit + 1 if query_id else limit
        )

        results = []
        for r in search_results:
            if r.get("filename"):
                # Exclude the query image itself from the results
                if query_id and r["id"] == query_id:
                    continue
                results.append({**r, "filename": self.base_folder / r["filename"]})

        logger.debug(f"Found {len(results)} results for query.")
        return results[:limit]

    # ---------------------------------------------------------------------
    def delete_file(self, image_path: Path):
        """
        Deletes a stored embedding from the vector store based on its file path.

        This is useful for removing images from the search index when they are
        deleted from the filesystem.

        Args:
            image_path: The absolute path to the image file whose embedding
                        should be deleted.
        """
        if not image_path.is_file():
            logger.warning(f"{image_path} is not a valid file.")
            return

        rel_path = self._relative_path(image_path)

        if not rel_path:
            logger.warning("Unmanaged file, can't be deleted")
            return

        point_id = self.make_id(rel_path)
        self.engine.delete_file(point_id)
