from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, HnswConfigDiff
from qdrant_client.models import VectorParams, Distance
from loguru import logger
from pathlib import Path
import hashlib


class QdrantImageStore:
    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        vector_size: int = 512,
        distance: Distance = Distance.COSINE,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 200,
        max_segment_size: int = 100000,
    ):
        """
        Initialize the Qdrant image vector store, creating the collection if missing.
        """
        self.collection_name = collection_name
        self.client = QdrantClient(url)

        vector_params = VectorParams(size=vector_size, distance=distance)
        hnsw_params = HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct)
        optimizer_params = {"max_segment_size": max_segment_size}

        if not self.client.collection_exists(collection_name=collection_name):
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params,
                hnsw_config=hnsw_params,
                optimizers_config=optimizer_params,
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists. Reusing it.")
            existing = self.client.get_collection(collection_name=collection_name)
            existing_params = existing.config.params.vectors
            if (
                existing_params.size != vector_params.size
                or existing_params.distance != vector_params.distance
            ):
                logger.error("Collection config differs from expected parameters!")
                logger.error(
                    f"Existing size: {existing_params.size}, distance: {existing_params.distance}"
                )
                logger.error(
                    f"Expected size: {vector_params.size}, distance: {vector_params.distance}"
                )
                raise ValueError("Collection config mismatch.")

    # ---------------------------------------------------------------------
    def _compute_id(self, rel_path: Path, base: Path | None = None) -> tuple[int, str]:
        """
        Compute a deterministic 63-bit integer ID and normalized path string.
        """
        if base:
            normalized = (
                rel_path.resolve().relative_to(base.resolve()).as_posix().lower()
            )
        else:
            normalized = rel_path.resolve().as_posix().lower()

        point_id = int(hashlib.sha1(normalized.encode()).hexdigest(), 16) % (2**63)
        return point_id, normalized

    # ---------------------------------------------------------------------
    def add_vector(self, rel_path: Path, vec_f32, base: Path | None = None):
        """
        Add or update a single image vector to Qdrant.
        """
        point_id, normalized = self._compute_id(rel_path, base)

        point = PointStruct(
            id=point_id,
            vector=vec_f32,
            payload={"filename": normalized},
        )

        self.client.upsert(collection_name=self.collection_name, points=[point])
        logger.debug(f"Upserted: {normalized} (ID={point_id})")

    # ---------------------------------------------------------------------
    def get_vector(self, rel_path: Path, base: Path | None = None):
        """
        Retrieve a point from Qdrant using the deterministic path-based ID.
        """
        point_id, _ = self._compute_id(rel_path, base)
        return self.client.retrieve(
            collection_name=self.collection_name, ids=[point_id]
        )

    # ---------------------------------------------------------------------
    def delete_vector(self, rel_path: Path, base: Path | None = None):
        """
        Delete a point based on its deterministic path ID.
        """
        point_id, normalized = self._compute_id(rel_path, base)
        self.client.delete(
            collection_name=self.collection_name, points_selector={"points": [point_id]}
        )
        logger.debug(f"Deleted: {normalized} (ID={point_id})")

    # ---------------------------------------------------------------------
    def search(self, query_vector, limit: int = 5, with_payload: bool = True):
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
            with_payload=with_payload,
        )

        formatted = [
            {
                "id": r.id,
                "score": r.score,
                "filename": r.payload.get("filename") if r.payload else None,
            }
            for r in results
        ]

        logger.debug(f"Search returned {len(formatted)} results.")
        return formatted
