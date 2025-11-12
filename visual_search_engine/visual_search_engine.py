from pathlib import Path
from typing import Optional, List, Union, Dict
import numpy as np

import hashlib
import time
from PIL import Image
import io
from qdrant_client.models import Distance

from .ml_inference import MLInference
from .store_interface import StoreInterface


class VisualSearchEngine:
    """
    Base class for visual search engines.

    Model-agnostic: does not assume a specific HEF file or vector type.
    Subclasses should provide appropriate model, collection_name, and dimensions.
    """

    def __init__(
        self,
        *,
        inference_engine: MLInference,
        store_interface: StoreInterface,
        logger=None,
        progress_bar_class=None,
    ):
        """Initialize both inference and vector store."""
        self.inference = inference_engine
        self.store = store_interface
        self.logger = logger
        self.progress_bar_class = progress_bar_class

    # ---------------------------------------------------------------------
    def _preprocess_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Loads and preprocesses an image from the given path.

        Args:
            path: The path to the image file.

        Returns:
            A NumPy array of the pre-processed image (RGB, resized, uint8),
            or None if the image cannot be loaded or processed.
        """
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(
                    self.inference.input_size, Image.LANCZOS
                )
                return np.array(img, dtype=np.uint8)
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"preprocess_image: Failed to load or preprocess {path}: {e}"
                )
            return None

    # ---------------------------------------------------------------------
    def add_file(self, id: int, path: Path, payload=None, force: bool = False) -> bool:
        """
        Computes and stores the embedding for a single image file.

        By default, this method will skip processing if an embedding for the
        given id already exists in the store.

        Args:
            id: The unique identifier for the image.
            path: The absolute path to the image file.
            force: If True, re-computes and updates the embedding even if it
                   already exists. Defaults to False.
        """
        if not path.is_file():
            if self.logger:
                self.logger.warning(f"{path} is not a valid file.")
            return False

        # --- Skip if embedding exists and force is False ---
        if not force:
            existing = self.store.get_vector(id)
            if existing:
                if self.logger:
                    self.logger.warning(f"Skipping {id}, embedding already exists.")
                return True

        # --- Preprocess image ---
        image_buffer = self._preprocess_image(path)
        if image_buffer is None:
            if self.logger:
                self.logger.warning(f"Skipping {path} due to preprocessing failure.")
            return False

        # --- Compute and store embedding ---
        vec_f32 = self.inference.infer(image_buffer, str(id))
        if vec_f32 is None:
            if self.logger:
                self.logger.warning(f"Failed to generate embedding for {id}")
            return False

        self.store.add_vector(id, vec_f32, payload=payload)
        if self.logger:
            self.logger.info(f"Added embedding for {id}")
        return True

    # ---------------------------------------------------------------------
    def _discover_and_filter_files(
        self, files: Dict[int, Path], force: bool
    ) -> Dict[int, Path]:

        if force:
            return files

        files_to_process: Dict[int, Path] = {}
        progress_bar = None
        if self.progress_bar_class:
            progress_bar = self.progress_bar_class(
                total_items=len(files),
                update_interval=25,
                message="Analyzing data files",
            )
        for i, id in enumerate(files.keys()):
            if not self.store.get_vector(id):
                files_to_process[id] = files[id]
            else:
                if self.logger:
                    self.logger.debug(f"Skipping {id}: already exists.")
            if progress_bar:
                progress_bar.update(i)
        if progress_bar:
            progress_bar.close(final_message="Finished successfully")

        if self.logger:
            self.logger.info(
                f"{len(files)} total images. {len(files_to_process)} need processing."
            )
        return files_to_process

    def _process_batch(
        self, buffers: Dict[str, np.ndarray], payload: Dict[int, Dict] = None
    ) -> int:
        """Process a batch of images and store their embeddings."""

        batch_results = self.inference.infer_batch(buffers)
        successful_embeddings = 0
        for id_str, vec_f32 in batch_results.items():
            if vec_f32 is not None:
                id = int(id_str)
                self.store.add_vector(
                    id,
                    vec_f32,
                    payload=payload.get(id, None) if payload else None,
                )
                successful_embeddings += 1

        return successful_embeddings

    def add_all(
        self,
        files: Dict[int, Path],
        force: bool = False,
        batch_size: int = 32,
        payload: Dict[int, Dict] = None,
    ):
        """
        Recursively finds and stores embeddings for all images in a directory using a greedy batching strategy.
        This method ensures that each batch sent to the inference engine is full, which is crucial for
        platforms that require fixed-size batches.
        """

        if self.logger:
            self.logger.info(f"Starting to index {len(files)} items")
        start_time = time.perf_counter()

        files_to_process = self._discover_and_filter_files(files, force)
        if not files_to_process:
            if self.logger:
                self.logger.info("No new images to process.")
            return

        total_successful_embeddings = 0
        total_images_attempted = 0
        batch_counter = 0

        # Accumulators for the greedy batching approach
        current_batch_buffers: Dict[str, np.ndarray] = {}
        progress_bar = None
        if self.progress_bar_class:
            progress_bar = self.progress_bar_class(
                total_items=len(files_to_process),
                update_interval=4,
                message="Processing images in batches",
            )
        additional_msg = ""
        for i, id in enumerate(files_to_process.keys()):
            total_images_attempted += 1
            image_buffer = self._preprocess_image(files_to_process[id])

            if image_buffer is not None:
                current_batch_buffers[str(id)] = image_buffer

                # If the batch is full, process it
                if len(current_batch_buffers) == batch_size:
                    batch_counter += 1
                    batch_start_time = time.perf_counter()

                    successful_in_batch = self._process_batch(
                        current_batch_buffers, payload
                    )
                    total_successful_embeddings += successful_in_batch

                    batch_time = time.perf_counter() - batch_start_time
                    avg_ms = (
                        (batch_time * 1000) / successful_in_batch
                        if successful_in_batch > 0
                        else 0
                    )

                    additional_msg = (
                        f"Inference batch #{batch_counter}  Avg: {avg_ms:.2f} ms/image"
                    )

                    # Reset accumulators for the next batch
                    current_batch_buffers = {}
            else:
                if self.logger:
                    self.logger.warning(f"Skipping {id} due to preprocessing failure.")
            if progress_bar:
                progress_bar.update(i, additional_msg, force=False)

        # Process any remaining images in the last, potentially partial, batch
        if current_batch_buffers:
            batch_counter += 1
            batch_start_time = time.perf_counter()

            successful_in_batch = self._process_batch(current_batch_buffers, payload)
            total_successful_embeddings += successful_in_batch

            batch_time = time.perf_counter() - batch_start_time
            avg_ms = (
                (batch_time * 1000) / successful_in_batch
                if successful_in_batch > 0
                else 0
            )

            if self.logger:
                self.logger.info(
                    f"Processed final batch #{batch_counter} ({successful_in_batch}/{len(current_batch_buffers)} successful) in {batch_time:.2f}s. Avg: {avg_ms:.2f} ms/image"
                )

        total_time = time.perf_counter() - start_time
        if progress_bar:
            progress_bar.close(
                final_message=f"Finished indexing. Processed {total_successful_embeddings} new embeddings from {total_images_attempted} attempted images in {total_time:.2f}s."
            )

    # ---------------------------------------------------------------------
    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        image_buffer = self._preprocess_image(image_path)
        if image_buffer is None:
            if self.logger:
                self.logger.warning(
                    f" Skipping embedding retrieval for {image_path} due to preprocessing failure."
                )
            return None

        return self.inference.infer(image_buffer, str(image_path))

    # ---------------------------------------------------------------------
    def search(self, query: Union[Path, np.ndarray], limit: int = 5) -> List[dict]:
        query_id = None
        if isinstance(query, Path):
            query_image_buffer = self._preprocess_image(query)
            if query_image_buffer is None:
                if self.logger:
                    self.logger.warning(f"Failed to preprocess query image {query}")
                return []
            query_vec = self.inference.infer(query_image_buffer, str(query))
            if query_vec is None:
                if self.logger:
                    self.logger.warning(f"Failed to compute embedding for {query}")
                return []
        else:
            query_vec = query

        search_results = self.store.search(
            query_vec, limit=limit + 1 if query_id else limit
        )

        if self.logger:
            self.logger.debug(f"Found {len(search_results)} results for query.")
        return search_results

    # ---------------------------------------------------------------------
    def delete_file(self, id: int):
        self.store.delete_vector(id)
        if self.logger:
            self.logger.debug(f"Deleted embedding for {id}")
