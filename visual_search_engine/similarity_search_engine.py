from pathlib import Path
from typing import Optional

from .ml_inference import MLInference
from .visual_search_engine import VisualSearchEngine
from .config import Config


class SimilaritySearchEngine(VisualSearchEngine):
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
        base_folder: Path,
        qdrant_url: str = Config.QDRANT_URL,
        vector_size: int = 512,
    ):
        """
        Initializes the SimilaritySearchEngine.

        This constructor sets up the search engine with default parameters
        suitable for the default model. Key parameters like the model path
        and vector size can be overridden to support different models.

        Args:
            inference_engine: An instance of MLInference to be used for embedding computation.
            base_folder: The base directory for image storage and relative path calculations.
            qdrant_url: The URL for the Qdrant service. Defaults to `Config.QDRANT_URL`.
            vector_size: The dimension of the vectors produced by the model. Defaults to 512.
        """
        super().__init__(
            inference_engine=inference_engine,
            base_folder=base_folder,
            collection_name="images2",
            vector_size=vector_size,
            qdrant_url=qdrant_url,
            distance_metric="COSINE",
        )
