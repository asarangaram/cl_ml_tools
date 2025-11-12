from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import numpy as np


class StoreInterface(ABC):
    """
    Abstract base class for a generic vector store interface.

    This class defines a standard interface for interacting with a vector store,
    allowing for different underlying implementations (e.g., Qdrant, Milvus, FAISS).
    Subclasses must implement methods for adding, retrieving, deleting, and searching vectors.
    """

    @abstractmethod
    def add_vector(self, id: int, vector: np.ndarray, payload: Optional[Dict] = None):
        """
        Adds a single vector to the store with a given ID and optional payload.
        """
        pass

    @abstractmethod
    def get_vector(self, id: int) -> Optional[List[Dict]]:
        """
        Retrieves a vector by its ID.
        """
        pass

    @abstractmethod
    def delete_vector(self, id: int):
        """
        Deletes a vector by its ID.
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, limit: int = 5) -> List[Dict]:
        """
        Searches for similar vectors in the store.
        """
        pass
