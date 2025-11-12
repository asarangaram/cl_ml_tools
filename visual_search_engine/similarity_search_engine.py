from pathlib import Path
from typing import Optional
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
        base_folder: Path,
        qdrant_url: str = Config.QDRANT_URL,
        profile_batch_size: int = 100,
        max_images: Optional[int] = None,
        hef_path: Path = Config.DEFAULT_HEF_PATH,
        vector_size: int = 512,
    ):
        """
        Initializes the SimilaritySearchEngine.

        This constructor sets up the search engine with default parameters
        suitable for the default model. Key parameters like the model path
        and vector size can be overridden to support different models.

        Args:
            base_folder: The base directory for image storage and relative path calculations.
            qdrant_url: The URL for the Qdrant service. Defaults to `Config.QDRANT_URL`.
            profile_batch_size: Batch size for profiling inference. Defaults to 100.
            max_images: Optional maximum number of images to process in a directory.
            hef_path: The path to the HEF model file. Defaults to `Config.DEFAULT_HEF_PATH`.
            vector_size: The dimension of the vectors produced by the model. Defaults to 512.
        """
        super().__init__(
            hef_path=hef_path,
            base_folder=base_folder,
            collection_name="images2",
            vector_size=vector_size,
            qdrant_url=qdrant_url,
            distance_metric="COSINE",
            profile_batch_size=profile_batch_size,
            max_images=max_images,
        )
