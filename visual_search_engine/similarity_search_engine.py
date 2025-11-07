from pathlib import Path
from typing import Optional
from .visual_search_engine import VisualSearchEngine


class SimilaritySearchEngine(VisualSearchEngine):
    """
    Specialized visual search engine for image embeddings
    using the local ResNet-18 HEF model.
    """

    def __init__(
        self,
        base_folder: Path,
        qdrant_url: str = "http://localhost:6333",
        profile_batch_size: int = 100,
        max_images: Optional[int] = 5000,
    ):
        hef_path = Path(__file__).parent / "models" / "resnet_v1_18_feature.hef"

        super().__init__(
            hef_path=hef_path,
            base_folder=base_folder,
            collection_name="images",
            vector_size=512,
            qdrant_url=qdrant_url,
            distance_metric="COSINE",
            profile_batch_size=profile_batch_size,
            max_images=max_images,
        )
