import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm
from pathlib import Path
from PIL import Image
import time
from typing import Callable, Optional
from .logger import logger


class HailoInference:
    """
    Hailo inference helper that computes embeddings for images.

    - Pure inference logic (no base/relative path handling)
    - Supports single-file or directory inference
    - Automatically detects image input size from HEF
    - Uses callbacks for directory processing
    - Normalizes embeddings before returning
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
        self, image_path: Path, config, bindings
    ) -> Optional[np.ndarray]:
        """Run inference on a single image and return normalized embedding."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB").resize(self.image_size, Image.LANCZOS)
                input_buffer = np.array(img, dtype=np.uint8)

            # Bind input/output buffers
            bindings.input().set_buffer(input_buffer)

            # Run synchronous inference
            config.run([bindings], timeout=self.timeout)
            vec = bindings.output().get_buffer()

            # Normalize vector
            vec_f32 = vec.astype("float32")
            norm = np.linalg.norm(vec_f32)
            if norm > 0:
                vec_f32 /= norm
            else:
                logger.warning(f"Zero norm vector for {image_path}")
                return None
            return vec_f32

        except Exception as e:
            logger.warning(f"[ERROR] Failed to process {image_path}: {e}")
            return None

    # ---------------------------------------------------------------------
    def process_file(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Process a single image file and return its embedding (vec_f32) or None.
        """
        if not image_path.is_file():
            logger.warning(f"process_file: {image_path} is not a valid file.")
            return None

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

                return self._process_image(image_path, config, bindings)

    # ---------------------------------------------------------------------
    def process_files(self, image_paths: list[Path], callback: Callable[[Path, np.ndarray], None]):
        """
        Process a list of image files and call the callback for each.

        Args:
            image_paths: A list of absolute paths to the image files.
            callback: A function to call for each successfully processed image,
                      receiving the image path and its embedding.
        """
        if not image_paths:
            return

        if callback is None:
            raise ValueError("Callback function must be provided.")

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

                total = len(image_paths)
                total_time = 0.0
                success_count = 0
                start_time = time.perf_counter()

                for i, image_path in enumerate(image_paths):
                    start = time.perf_counter()
                    vec_f32 = self._process_image(image_path, config, bindings)
                    if vec_f32 is not None:
                        callback(image_path, vec_f32)
                        success_count += 1
                    total_time += time.perf_counter() - start
                    last = i == len(image_paths) - 1
                    if (success_count % self.profile_batch_size) == 0 or last:
                        avg_ms = (total_time * 1000) / success_count if success_count > 0 else 0
                        error_count = (i + 1) - success_count
                        elapsed = time.perf_counter() - start_time
                        logger.info(
                            f"{elapsed:.2f}s elapsed: processed {success_count}/{total}, "
                            f"avg {avg_ms:.2f} ms/image "
                            f"{'(errors: ' + str(error_count) + ')' if error_count > 0 else ''}"
                        )
