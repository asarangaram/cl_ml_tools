from pathlib import Path
from typing import Callable, Optional, List, Union, Dict
import numpy as np
import time

from .ml_inference import MLInference
from .store_interface import StoreInterface

FileInput = Union[Path, bytes]


class VectorCore:
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
        preprocess_cb: Optional[Callable[[FileInput], Optional[np.ndarray]]] = None,
    ):
        """Initialize both inference and vector store."""
        self.inference = inference_engine
        self.store = store_interface
        self.logger = logger
        self.progress_bar_class = progress_bar_class
        self.preprocess_cb = preprocess_cb

    # ---------------------------------------------------------------------
    def load_to_buffer(self, file_input: FileInput) -> Optional[np.ndarray]:
        """Load Path or bytes into a bytes buffer"""
        if isinstance(file_input, Path):
            try:
                with open(file_input, "rb") as f:
                    return np.array(f.read(), dtype=np.uint8)
            except Exception as e:
                print(f"Failed to read file {file_input}: {e}")
                return None
        elif isinstance(file_input, bytes):
            return np.array(file_input, dtype=np.uint8)
        else:
            raise TypeError(f"Unsupported input type: {type(file_input)}")

    # ---------------------------------------------------------------------
    def add_file(
        self, id: int, data: FileInput, payload=None, force: bool = False
    ) -> bool:
        """
        Computes and stores the embedding for a single image file.

        By default, this method will skip processing if an embedding for the
        given id already exists in the store.

        Args:
            id: The unique identifier for the image.
            data: The file input, which can be a Path object or bytes.
            force: If True, re-computes and updates the embedding even if it
                   already exists. Defaults to False.
        """
        # --- Skip if embedding exists and force is False ---
        if not force:
            existing = self.store.get_vector(id)
            if existing:
                if self.logger:
                    self.logger.warning(f"Skipping {id}, embedding already exists.")
                return True

        if payload is None:
            if self.logger:
                self.logger.info(f"Skipping {id}, Payload is missing")
            return False

        buffer = (
            self.preprocess_cb(data)
            if self.preprocess_cb
            else self.load_to_buffer(data)
        )
        if buffer is None:
            self.logger.warning(f"add_file: Failed to preprocess {id}")
            return False

        # --- Compute and store embedding ---
        vec_f32 = self.inference.infer(buffer, str(id))
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
        self, files: Dict[int, FileInput], force: bool
    ) -> Dict[int, FileInput]:
        """
        Discovers and filters files that need to be processed.

        Args:
            files: A dictionary of files to process.
            force: If True, re-processes all files even if they already exist.

        Returns:
            A dictionary of files that need to be processed.
        """

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
                curr_payload = payload.get(id, None) if payload else None
                if curr_payload is None:
                    if self.logger:
                        self.logger.info(f"Skipping {id}, Payload is missing")
                    return False
                else:
                    self.store.add_vector(
                        id,
                        vec_f32,
                        payload=curr_payload,
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
        Computes and stores embeddings for a dictionary of files, using a greedy batching strategy.

        This method processes files in batches to optimize inference performance. It can skip files
        that already have embeddings unless 'force' is set to True.

        Args:
            files: A dictionary mapping a unique ID to the file input (Path or bytes).
            force: If True, re-computes and updates embeddings even if they already exist.
                   Defaults to False.
            batch_size: The number of items to process in a single inference batch.
                        Defaults to 32.
            payload: A dictionary mapping the file ID to a payload dictionary to be stored
                     alongside the vector.
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
            data = files_to_process[id]
            buffer = (
                self.preprocess_cb(data)
                if self.preprocess_cb
                else self.load_to_buffer(data)
            )

            if buffer is not None:
                current_batch_buffers[str(id)] = buffer

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
    def delete_file(self, id: int):
        """
        Deletes the embedding for a single file.

        Args:
            id: The unique identifier for the file.
        """
        self.store.delete_vector(id)
        if self.logger:
            self.logger.debug(f"Deleted embedding for {id}")

    # ---------------------------------------------------------------------
    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Computes the embedding for a single image file.

        Args:
            image_path: The absolute path to the image file.

        Returns:
            A numpy array representing the embedding, or None if it cannot be computed.
        """
        buffer = (
            self.preprocess_cb(image_path)
            if self.preprocess_cb
            else self.load_to_buffer(image_path)
        )
        if buffer is None:
            return None

        return self.inference.infer(buffer, "Unknown")

    # ---------------------------------------------------------------------
    def search(self, data: FileInput, limit: int = 5) -> Optional[List[dict]]:
        """
        Searches for similar images in the store.

        Args:
            data: The image data to search for.
            limit: The maximum number of results to return.

        Returns:
            A list of search results, or None if an error occurs.
        """
        query_id = None
        buffer = (
            self.preprocess_cb(data)
            if self.preprocess_cb
            else self.load_to_buffer(data)
        )
        query_vec = self.inference.infer(buffer, "Unknown")
        if query_vec is None:
            return None

        search_results = self.store.search(
            query_vec, limit=limit + 1 if query_id else limit
        )

        if self.logger:
            self.logger.debug(f"Found {len(search_results)} results for query.")
        return search_results
