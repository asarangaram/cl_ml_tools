import os
import sys

from pathlib import Path
from typing import Optional, List, Union, Dict
import numpy as np
import hashlib
from PIL import Image
from visual_search_engine.visual_search_engine import VisualSearchEngine, FileInput
from .inference.hailo_inference import HailoInference
from .store.qdrant_image_store import QdrantImageStore
from .progress_bar.progress_bar import ProgressBar

from qdrant_client.models import Distance

from loguru import logger


class Config:
    """
    Configuration class for the application.
    """

    # --- Qdrant Configuration ---
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    # --- Logging Configuration ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Model Configuration ---
    # Default HEF model path for the SimilaritySearchEngine
    DEFAULT_HEF_PATH: Path = (
        Path(__file__).parent / "inference" / "models" / "resnet_v1_18_feature.hef"
    )


logger.remove()
logger.add(sys.stderr, level=Config.LOG_LEVEL)


class SimilaritySearch:
    """
    Base class for visual search engines.

    Model-agnostic: does not assume a specific HEF file or vector type.
    Subclasses should provide appropriate model, collection_name, and dimensions.
    """

    def __init__(
        self,
        base_folder: Path,
    ):
        """Initialize both inference and vector store."""
        self.base_folder = Path(base_folder).resolve()
        # Initialize HailoInference
        self.hailo_inference_engine = HailoInference(
            hef_path=Config.DEFAULT_HEF_PATH,
            profile_batch_size=100,
            max_items=None,
            logger=logger,
        )

        # Initialize QdrantImageStore
        self.qdrant_store = QdrantImageStore(
            collection_name="images2",
            url=Config.QDRANT_URL,
            vector_size=512,  # Default vector size for the model
            distance=Distance.COSINE,
            logger=logger,
        )
        self.engine = VisualSearchEngine(
            inference_engine=self.hailo_inference_engine,
            store_interface=self.qdrant_store,
            logger=logger,
            progress_bar_class=ProgressBar,
            preprocess_cb=self.preprocess,
            # preprocess_cb: Optional[Callable[[FileInput], Optional[np.ndarray]]] = None,
        )
        self.logger = logger

    def preprocess(self, data: FileInput) -> Optional[np.ndarray]:
        if isinstance(data, Path):
            if not data.is_file():
                if logger:
                    logger.warning(f"{data} is not a valid file.")
                return None
            try:
                with Image.open(data) as img:
                    img = img.convert("RGB").resize(
                        self.hailo_inference_engine.input_size, Image.LANCZOS
                    )
                    return np.array(img, dtype=np.uint8)
            except Exception as e:
                return None
        else:
            return np.array(data, dtype=np.uint8)

    # ---------------------------------------------------------------------
    @staticmethod
    def make_id(path: Union[str, Path]) -> int:
        """Compute a deterministic ID from a file path."""
        try:
            normalized = str(path).strip().replace("\\", "/").lower()
            return int(hashlib.sha1(normalized.encode()).hexdigest(), 16) % (2**63)
        except ValueError:
            return None

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
            if self.logger:
                self.logger.warning(f"{image_path} is not a valid file.")
            return

        point_id = self.make_id(image_path)
        if point_id:
            return self.engine.add_file(id=point_id, data=image_path, force=force)
        else:
            return False

    def add_dir(self, dir_path: Path, force: bool = False, batch_size: int = 32):
        if not dir_path.is_dir():
            if self.logger:
                self.logger.warning(f"{dir_path} is not a valid directory.")
            return

        if not self._relative_path(dir_path):
            if self.logger:
                self.logger.warning(f"{dir_path} is not a managed directory")
            return

        all_files = [
            p for ext in ("*.jpg", "*.jpeg", "*.png") for p in dir_path.rglob(ext)
        ]

        if not all_files:
            if self.logger:
                self.logger.info("No new images to process.")
            return

        files: Dict[int, Path] = {}
        payloads: Dict[int, Dict] = {}
        for file in all_files:
            id = self.make_id(file)
            if id:
                files[id] = file
                payloads[id] = {"filename": str(self._relative_path(file))}

        self.engine.add_all(
            files=files, payload=payloads, force=force, batch_size=batch_size
        )

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
            query_id = self.make_id(query)

        search_results = self.engine.search(
            query, limit=limit + 1 if query_id else limit
        )
        print(search_results)
        results = []
        for r in search_results:
            if r.get("filename"):
                # Exclude the query image itself from the results
                if query_id and r["id"] == query_id:
                    continue
                results.append({**r, "filename": self.base_folder / r["filename"]})
            else:
                # raise Exception("Failed to get payload when searching")
                pass

        if logger:
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
            if self.logger:
                self.logger.warning(f"{image_path} is not a valid file.")
            return

        point_id = self.make_id(image_path)

        if not point_id:
            if self.logger:
                self.logger.warning("Unmanaged file, can't be deleted")
            return

        self.engine.delete_file(point_id)
