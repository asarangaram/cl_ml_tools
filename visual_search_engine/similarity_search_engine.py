from pathlib import Path
from typing import Optional

from .ml_inference import MLInference
from .visual_search_engine import VisualSearchEngineFileSystem
from ..config import Config
from .store_interface import StoreInterface


class SimilaritySearchEngineFileSystem(VisualSearchEngineFileSystem):
    """
    A specialized visual search engine with sensible defaults for image embeddings.

    This class provides a convenient starting point for visual search by
    pre-configuring the VisualSearchEngine with common settings for image
    similarity, such as a default model and COSINE distance. The model and
    its vector size can be easily overridden.
    """

    def __init__(
        self,
        inference_engine: MLInference,
        store_interface: StoreInterface,
        base_folder: Path,
        logger=None,
        progress_bar_class=None,
    ):
        """
        Initializes the SimilaritySearchEngine.

        This constructor sets up the search engine with default parameters
        suitable for the default model. Key parameters like the model path
        and vector size can be overridden to support different models.

        Args:
            inference_engine: An instance of MLInference to be used for embedding computation.
            store_interface: An instance of StoreInterface to be used for vector storage.
            base_folder: The base directory for image storage and relative path calculations.
            logger: Optional logger instance for logging messages.
            progress_bar_class: Optional class for progress bar.
        """
        super().__init__(
            inference_engine=inference_engine,
            store_interface=store_interface,
            base_folder=base_folder,
            logger=logger,
            progress_bar_class=progress_bar_class,
        )
