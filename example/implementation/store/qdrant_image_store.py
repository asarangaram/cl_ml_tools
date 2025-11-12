from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, HnswConfigDiff
from qdrant_client.models import VectorParams, Distance

import numpy as np
from visual_search_engine.store_interface import StoreInterface


class QdrantImageStore(StoreInterface):
    """
    Manages image vectors in a Qdrant collection.

    This class provides an interface to interact with Qdrant, handling
    collection creation, adding new image embeddings, retrieving, deleting,
    and performing similarity searches. It ensures that the Qdrant collection
    is properly configured for efficient vector storage and retrieval.
    """

    def __init__(
        self,
        collection_name: str,
        url: str,
        vector_size: int = 512,
        distance: Distance = Distance.COSINE,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        max_segment_size: int = 100000,
        logger=None,
    ):
        """
        Initialize the Qdrant image vector store, creating the collection if missing.
        """
        self.collection_name = collection_name
        self.client = QdrantClient(url)
        self.logger = logger

        vector_params = VectorParams(size=vector_size, distance=distance)
        hnsw_params = HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct)
        optimizer_params = {"max_segment_size": max_segment_size}

        if not self.client.collection_exists(collection_name=collection_name):
            if self.logger:
                self.logger.debug(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params,
                hnsw_config=hnsw_params,
                optimizers_config=optimizer_params,
            )
        else:
            if self.logger:
                self.logger.debug(
                    f"Collection '{collection_name}' already exists. Reusing it."
                )
            existing = self.client.get_collection(collection_name=collection_name)
            existing_params = existing.config.params.vectors
            if (
                existing_params.size != vector_params.size
                or existing_params.distance.value != vector_params.distance.value
            ):
                if self.logger:
                    self.logger.error(
                        "Collection config differs from expected parameters!"
                    )
                    self.logger.error(
                        f"Existing size: {existing_params.size}, distance: {existing_params.distance}"
                    )
                    self.logger.error(
                        f"Expected size: {vector_params.size}, distance: {vector_params.distance}"
                    )
                raise ValueError("Collection config mismatch.")

    # ---------------------------------------------------------------------
    def add_vector(
        self, point_id: int, vec_f32: np.ndarray, payload: Optional[dict] = None
    ):
        """
        Add or update a single image vector to Qdrant.
        """

        point = PointStruct(
            id=point_id,
            vector=vec_f32,
            payload=payload,
        )

        self.client.upsert(collection_name=self.collection_name, points=[point])
        if self.logger:
            self.logger.debug(f"Upserted: {point_id} ")

    # ---------------------------------------------------------------------
    def get_vector(self, point_id: int):
        """
        Retrieve a point from Qdrant using the deterministic path-based ID.
        """

        return self.client.retrieve(
            collection_name=self.collection_name, ids=[point_id]
        )

    # ---------------------------------------------------------------------
    def delete_vector(self, point_id: int):
        """
        Delete a point based on its deterministic path-based ID.
        """
        self.client.delete(
            collection_name=self.collection_name, points_selector={"points": [point_id]}
        )
        if self.logger:
            self.logger.debug(f"Deleted: {point_id}")

    # ---------------------------------------------------------------------
    def search(
        self,
        query_vector,
        limit: int = 5,
        with_payload: bool = True,
        score_threshold=0.85,
    ):
        """
        Search for similar vectors in the collection.

        Args:
            query_vector: The query embedding (float32 list or numpy array)
            limit: Number of nearest neighbors to return
            with_payload: Whether to return payload (metadata) along with results

        Returns:
            List of search results with (id, score, payload)
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )

        formatted = []
        for r in results:
            point_data = {"id": r.id, "score": r.score}
            if r.payload:
                point_data.update(r.payload)
            formatted.append(point_data)

        if self.logger:
            self.logger.debug(f"Search returned {len(formatted)} results.")
        return formatted
