from pathlib import Path
from typing import Optional, List, Union, Dict
import numpy as np

from .progress_bar import ProgressBar
from .logger import logger
import hashlib
import time
from PIL import Image
import io
from qdrant_client.models import Distance

from .hailo_inference import HailoInference
from .qdrant_image_store import QdrantImageStore
from .config import Config


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
        qdrant_url: str = Config.QDRANT_URL,
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
    def _relative_path(self, path: Path) -> Optional[Path]:
        """Compute path relative to base folder, or absolute if outside."""
        try:
            return path.resolve().relative_to(self.base_folder)
        except ValueError:
            return None

    # ---------------------------------------------------------------------
    def _preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Loads and preprocesses an image from the given path.

        Args:
            image_path: The path to the image file.

        Returns:
            A NumPy array of the pre-processed image (RGB, resized, uint8),
            or None if the image cannot be loaded or processed.
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB").resize(
                    self.inference.image_size, Image.LANCZOS
                )
                return np.array(img, dtype=np.uint8)
        except Exception as e:
            logger.warning(
                f"preprocess_image: Failed to load or preprocess {image_path}: {e}"
            )
            return None

    # ---------------------------------------------------------------------
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

            # --- Skip if embedding exists and force is False ---
            if not force:
                existing = self.store.get_vector(point_id)
                if existing:
                    logger.debug(f"Skipping {rel_path}, embedding already exists.")
                    return

            # --- Preprocess image ---
            image_buffer = self._preprocess_image(image_path)
            if image_buffer is None:
                logger.warning(f"Skipping {image_path} due to preprocessing failure.")
                return

            # --- Compute and store embedding ---
            vec_f32 = self.inference.process_file(image_buffer, image_path)
            if vec_f32 is None:
                logger.warning(f"Failed to generate embedding for {image_path}")
                return

            self.store.add_vector(
                point_id, vec_f32, payload={"filename": str(rel_path)}
            )
            logger.debug(f"Added embedding for {rel_path}")
            return True
        else:
            return False

    # ---------------------------------------------------------------------
    def _discover_and_filter_files(self, dir_path: Path, force: bool) -> List[Path]:
        """Discover and filter image files in a directory."""
        all_files = [
            p for ext in ("*.jpg", "*.jpeg", "*.png") for p in dir_path.rglob(ext)
        ]
        if not all_files:
            logger.warning(f"No image files found in directory: {dir_path}")
            return []

        if force:
            return all_files

        files_to_process = []
        progress_bar = ProgressBar(
            total_items=len(all_files),
            update_interval=25,
            message="Analyzing data files",
        )
        for i, f in enumerate(all_files):
            # logger.info(f"checkinn {i}")
            rel_path = self._relative_path(f)
            if rel_path:
                point_id = self.make_id(rel_path)
                if not self.store.get_vector(point_id):
                    files_to_process.append(f)
                else:
                    logger.debug(f"Skipping {f}: already exists.")
            progress_bar.update(i)
        progress_bar.close(final_message="Finished successfully")

        logger.info(
            f"Found {len(all_files)} total images. {len(files_to_process)} need processing."
        )
        return files_to_process

    def _process_batch(self, buffers: List[np.ndarray], paths: List[Path]) -> int:
        """Process a batch of images and store their embeddings."""
        batch_results = self.inference.process_files(buffers, paths)
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
        """
        Recursively finds and stores embeddings for all images in a directory using a greedy batching strategy.
        This method ensures that each batch sent to the inference engine is full, which is crucial for
        platforms that require fixed-size batches.
        """
        if not dir_path.is_dir():
            logger.warning(f"{dir_path} is not a valid directory.")
            return

        if not self._relative_path(dir_path):
            logger.warning(f"{dir_path} is not a managed directory")
            return

        logger.info(f"Starting to index directory: {dir_path}")
        start_time = time.perf_counter()

        files_to_process = self._discover_and_filter_files(dir_path, force)
        if not files_to_process:
            logger.info("No new images to process.")
            return

        total_successful_embeddings = 0
        total_images_attempted = 0
        batch_counter = 0

        # Accumulators for the greedy batching approach
        current_batch_buffers: List[np.ndarray] = []
        current_batch_paths: List[Path] = []
        progress_bar = ProgressBar(
            total_items=len(files_to_process),
            update_interval=4,
            message="Processing images in batches",
        )
        additional_msg = ""
        for i, f in enumerate(files_to_process):
            total_images_attempted += 1
            image_buffer = self._preprocess_image(f)

            if image_buffer is not None:
                current_batch_buffers.append(image_buffer)
                current_batch_paths.append(f)

                # If the batch is full, process it
                if len(current_batch_buffers) == batch_size:
                    batch_counter += 1
                    batch_start_time = time.perf_counter()

                    successful_in_batch = self._process_batch(
                        current_batch_buffers, current_batch_paths
                    )
                    total_successful_embeddings += successful_in_batch

                    batch_time = time.perf_counter() - batch_start_time
                    avg_ms = (
                        (batch_time * 1000) / successful_in_batch
                        if successful_in_batch > 0
                        else 0
                    )

                    # logger.info(
                    #    f"Processed batch #{batch_counter} ({successful_in_batch}/{len(current_batch_buffers)} successful) in {batch_time:.2f}s. Avg: {avg_ms:.2f} ms/image"
                    # )
                    additional_msg = (
                        f"Inference batch #{batch_counter}  Avg: {avg_ms:.2f} ms/image"
                    )

                    # Reset accumulators for the next batch
                    current_batch_buffers = []
                    current_batch_paths = []
            else:
                logger.warning(f"Skipping {f} due to preprocessing failure.")
            progress_bar.update(i, additional_msg, force=False)

        # Process any remaining images in the last, potentially partial, batch
        if current_batch_buffers:
            batch_counter += 1
            batch_start_time = time.perf_counter()

            successful_in_batch = self._process_batch(
                current_batch_buffers, current_batch_paths
            )
            total_successful_embeddings += successful_in_batch

            batch_time = time.perf_counter() - batch_start_time
            avg_ms = (
                (batch_time * 1000) / successful_in_batch
                if successful_in_batch > 0
                else 0
            )

            logger.info(
                f"Processed final batch #{batch_counter} ({successful_in_batch}/{len(current_batch_buffers)} successful) in {batch_time:.2f}s. Avg: {avg_ms:.2f} ms/image"
            )

        total_time = time.perf_counter() - start_time
        logger.info(
            f"Finished indexing. Processed {total_successful_embeddings} new embeddings from {total_images_attempted} attempted images in {total_time:.2f}s."
        )
        progress_bar.close(final_message="Finished successfully")

    # ---------------------------------------------------------------------
    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Retrieves the embedding for a given image.

        This method first checks if the embedding exists in the vector store.
        If the image is part of the indexed `base_folder`, it looks up the
        embedding there. If it's not found or the image is outside the
        `base_folder`, it computes the embedding on the fly.

        Args:
            image_path: The absolute path to the image file.

        Returns:
            A numpy array representing the image embedding, or None if
            the embedding cannot be computed.
        """
        rel_path = self._relative_path(image_path)

        if rel_path:
            point_id = self.make_id(rel_path)
            point = self.store.get_vector(point_id)
            if point and hasattr(point[0], "vector") and point[0].vector is not None:
                return np.array(point[0].vector, dtype=np.float32)
            logger.debug(f"Embedding not found in store for {rel_path}, computing...")

        # --- Preprocess image ---
        image_buffer = self._preprocess_image(image_path)
        if image_buffer is None:
            logger.warning(
                f" Skipping embedding retrieval for {image_path} due to preprocessing failure."
            )
            return None

        return self.inference.process_file(image_buffer, image_path)

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

            query_image_buffer = self._preprocess_image(query)
            if query_image_buffer is None:
                logger.warning(f"Failed to preprocess query image {query}")
                return []
            query_vec = self.inference.process_file(query_image_buffer, query)
            if query_vec is None:
                logger.warning(f"Failed to compute embedding for {query}")
                return []
        else:
            query_vec = query

        search_results = self.store.search(
            query_vec, limit=limit + 1 if query_id else limit
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
        self.store.delete_vector(point_id)
        logger.debug(f"Deleted embedding for {rel_path}")
