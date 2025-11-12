import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm
from pathlib import Path
import time
from typing import Optional, Dict, List
from .logger import logger


class HailoInference:
    """
    Hailo inference helper that computes embeddings for images.

    This class is responsible for pure inference logic, accepting pre-processed
    image buffers and returning embeddings. It automatically detects image
    input size from the HEF model and normalizes embeddings.
    """

    def __init__(
        self,
        hef_path: Path,
        profile_batch_size: int = 100,
        max_images: Optional[int] = None,
        timeout: int = 1000,
    ):
        self.hef_path = Path(hef_path)
        self.profile_batch_size = profile_batch_size
        self.max_images = max_images
        self.timeout = timeout

        # --- Automatically detect image input size from HEF ---
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

            image_size = (w, h)
            logger.debug(f"Model expects input image size: {image_size}")

        self.image_size = image_size

    # ---------------------------------------------------------------------
    def _process_image(
        self, input_buffer: np.ndarray, config, bindings, image_path: Path
    ) -> Optional[np.ndarray]:
        """Run inference on a single pre-processed image buffer and return normalized embedding."""
        try:
            bindings.input().set_buffer(input_buffer)
            config.run([bindings], timeout=self.timeout)
            vec = bindings.output().get_buffer()

            vec_f32 = vec.astype("float32")
            norm = np.linalg.norm(vec_f32)
            if norm > 0:
                vec_f32 /= norm
            else:
                logger.warning(f"Zero norm vector for {image_path}")
                return None
            return vec_f32
        except (ValueError, RuntimeError) as e:
            logger.warning(f"[ERROR] Failed to process {image_path} due to {type(e).__name__}: {e}")
            return None
        except Exception as e:
            logger.critical(f"[CRITICAL] An unexpected error occurred while processing {image_path}: {e}")
            return None

    # ---------------------------------------------------------------------
    def process_file(self, image_buffer: np.ndarray, image_path: Path) -> Optional[np.ndarray]:
        """
        Process a single pre-processed image buffer and return its embedding (vec_f32) or None.

        Args:
            image_buffer: The pre-processed image as a NumPy array.
            image_path: The original path of the image (for logging).

        Returns:
            A numpy array representing the image embedding, or None if
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

                return self._process_image(image_buffer, config, bindings, image_path)

    # ---------------------------------------------------------------------
    def process_files(
        self, image_buffers: List[np.ndarray], image_paths: List[Path]
    ) -> Dict[Path, Optional[np.ndarray]]:
        """
        Process a list of pre-processed image buffers and return their embeddings.

        Args:
            image_buffers: A list of pre-processed images as NumPy arrays.
            image_paths: A list of original absolute paths corresponding to the image buffers.

        Returns:
            A dictionary mapping each image path to its computed embedding (NumPy array),
            or None if the embedding could not be computed for that image.
        """
        if not image_buffers:
            return {}

        if len(image_buffers) != len(image_paths):
            raise ValueError("Length of image_buffers and image_paths must be the same.")

        results: Dict[Path, Optional[np.ndarray]] = {}
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

                for i, input_buffer in enumerate(image_buffers):
                    image_path = image_paths[i]
                    vec_f32 = self._process_image(input_buffer, config, bindings, image_path)
                    results[image_path] = vec_f32
        return results
