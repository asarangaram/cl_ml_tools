import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm
from pathlib import Path
import time
from typing import Optional, Dict, List, str
from .logger import logger


class HailoInference:
    """
    Hailo inference helper that computes embeddings for input data.

    This class is responsible for pure inference logic, accepting pre-processed
    buffers and returning embeddings. It automatically detects input
    size from the HEF model and normalizes embeddings.
    """

    def __init__(
        self,
        hef_path: Path,
        profile_batch_size: int = 100,
        max_items: Optional[int] = None,
        timeout: int = 1000,
    ):
        self.hef_path = Path(hef_path)
        self.profile_batch_size = profile_batch_size
        self.max_items = max_items
        self.timeout = timeout

        # --- Automatically detect input size from HEF ---
        logger.debug(f"Auto-detecting input size from HEF: {self.hef_path}")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        with VDevice(params) as vdevice:
            infer_model = vdevice.create_infer_model(str(self.hef_path))
            shape = infer_model.input().shape
            logger.debug(f"Detected input shape: {shape}")

            # Extract HxW from shape (supports NCHW, NHWC, HWC, CHW)
            if len(shape) == 4:  # NCHW or NHWC
                if shape[1] == 3:  # NCHW (e.g., 1, 3, 224, 224)
                    h, w = shape[2], shape[3]
                else:  # NHWC (e.g., 1, 224, 224, 3)
                    h, w = shape[1], shape[2]
            elif len(shape) == 3:  # HWC or CHW
                if shape[0] == 3:  # CHW (e.g., 3, 224, 224)
                    h, w = shape[1], shape[2]
                else:  # HWC (e.g., 224, 224, 3)
                    h, w = shape[0], shape[1]
            else:
                raise ValueError(f"Unsupported input shape: {shape}")

            input_size = (w, h)
            logger.debug(f"Model expects input size: {input_size}")

        self.input_size = input_size

    # ---------------------------------------------------------------------
    def _process_buffer(
        self, input_buffer: np.ndarray, config, bindings, label: str
    ) -> Optional[np.ndarray]:
        """Run inference on a single pre-processed buffer and return normalized embedding."""
        try:
            bindings.input().set_buffer(input_buffer)
            config.run([bindings], timeout=self.timeout)
            vec = bindings.output().get_buffer()

            vec_f32 = vec.astype("float32")
            norm = np.linalg.norm(vec_f32)
            if norm > 0:
                vec_f32 /= norm
            else:
                logger.warning(f"Zero norm vector for {label}")
                return None
            return vec_f32
        except (ValueError, RuntimeError) as e:
            logger.warning(f"[ERROR] Failed to process {label} due to {type(e).__name__}: {e}")
            return None
        except Exception as e:
            logger.critical(f"[CRITICAL] An unexpected error occurred while processing {label}: {e}")
            return None

    # ---------------------------------------------------------------------
    def infer(self, buffer: np.ndarray, label: str) -> Optional[np.ndarray]:
        """
        Process a single pre-processed buffer and return its embedding (vec_f32) or None.

        Args:
            buffer: The pre-processed data as a NumPy array.
            label: The label of the data (for logging).

        Returns:
            A numpy array representing the embedding, or None if
            the embedding cannot be computed.
        """
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        with VDevice(params) as vdevice:
            infer_model = vdevice.create_infer_model(str(self.hef_path))
            with infer_model.configure() as config:
                bindings = config.create_bindings()
                output_buffer = np.empty(
                    list(infer_model.output().shape), dtype=np.uint8
                )
                bindings.output().set_buffer(output_buffer)

                return self._process_buffer(buffer, config, bindings, label)

    # ---------------------------------------------------------------------
    def infer_batch(
        self, buffers: Dict[str, np.ndarray]
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Process a dictionary of pre-processed buffers and return their embeddings.

        Args:
            buffers: A dictionary where keys are string labels and values are pre-processed data as NumPy arrays.

        Returns:
            A dictionary mapping each label to its computed embedding (NumPy array),
            or None if the embedding could not be computed for that data.
        """
        if not buffers:
            return {}

        results: Dict[str, Optional[np.ndarray]] = {}
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        with VDevice(params) as vdevice:
            infer_model = vdevice.create_infer_model(str(self.hef_path))
            with infer_model.configure() as config:
                bindings = config.create_bindings()
                output_buffer = np.empty(
                    list(infer_model.output().shape), dtype=np.uint8
                )
                bindings.output().set_buffer(output_buffer)

                for label, input_buffer in buffers.items():
                    vec_f32 = self._process_buffer(input_buffer, config, bindings, label)
                    results[label] = vec_f32
        return results
