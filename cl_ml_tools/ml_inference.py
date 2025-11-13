from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np


class MLInference(ABC):
    """
    Abstract base class for a generic machine learning inference engine.

    This class defines a standard interface for performing inference on single
    data buffers or batches of buffers. Subclasses must implement the `infer`
    and `infer_batch` methods, as well as the `input_size` property.
    """

    @property
    @abstractmethod
    def input_size(self) -> tuple[int, int]:
        """
        Returns the expected input size (width, height) for the model.
        """
        pass

    @abstractmethod
    def infer(self, buffer: np.ndarray, label: str) -> Optional[np.ndarray]:
        """
        Process a single pre-processed buffer and return its embedding.

        Args:
            buffer: The pre-processed data as a NumPy array.
            label: The label of the data (for logging).

        Returns:
            A numpy array representing the embedding, or None if it cannot be computed.
        """
        pass

    @abstractmethod
    def infer_batch(
        self, buffers: Dict[str, np.ndarray]
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Process a dictionary of pre-processed buffers and return their embeddings.

        Args:
            buffers: A dictionary where keys are string labels and values are
                     pre-processed data as NumPy arrays.

        Returns:
            A dictionary mapping each label to its computed embedding, or None.
        """
        pass
