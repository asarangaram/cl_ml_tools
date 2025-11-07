import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm
from pathlib import Path
from PIL import Image
import time
from typing import Callable, Optional
from loguru import logger


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
        max_images: Optional[int] = 5000,
    ):
        self.hef_path = Path(hef_path)
        self.profile_batch_size = profile_batch_size
        self.max_images = max_images

        # --- Automatically detect image input size from HEF ---
        logger.info(f"Auto-detecting input size from HEF: {self.hef_path}")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        with VDevice(params) as vdevice:
            infer_model = vdevice.create_infer_model(str(self.hef_path))
            shape = infer_model.input().shape
            logger.debug(f"Detected input shape: {shape}")

            # Extract H×W from shape (supports NCHW or NHWC)
            if len(shape) == 4:  # e.g. (1, 224, 224, 3)
                h, w = shape[1], shape[0]
                raise ValueError(f"Recheck shape for batch mode: {shape}")
            elif len(shape) == 3:  # e.g. (224, 224, 3)
                h, w = shape[1], shape[0]
            else:
                raise ValueError(f"Unexpected input shape: {shape}")

            image_size = (w, h)
            logger.info(f"Model expects input image size: {image_size}")

        self.image_size = image_size

    # ---------------------------------------------------------------------
    def _process_image(
        self, image_path: Path, config, output_buffer
    ) -> Optional[np.ndarray]:
        """Run inference on a single image and return normalized embedding."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB").resize(self.image_size, Image.LANCZOS)
                input_buffer = np.asarray(img, dtype=np.uint8)

            bindings = config.create_bindings()
            # Bind input/output buffers
            bindings.input().set_buffer(input_buffer)
            bindings.output().set_buffer(output_buffer)

            # Run synchronous inference
            config.run([bindings])
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

                output_buffer = np.empty(
                    list(infer_model.output().shape), dtype=np.uint8
                )
                return self._process_image(image_path, config, output_buffer)

    # ---------------------------------------------------------------------
    def process_dir(self, dir_path: Path, callback: Callable[[Path, np.ndarray], None]):
        """
        Process all images in a directory (recursively) and call the callback
        for each (image_path, vec_f32).
        """
        if not dir_path.is_dir():
            logger.warning(f"process_dir: {dir_path} is not a valid directory.")
            return

        if callback is None:
            raise ValueError("Callback function must be provided.")

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        with VDevice(params) as vdevice:
            infer_model = vdevice.create_infer_model(self.hef_path)
            with infer_model.configure() as config:
                bindings = config.create_bindings()
                output_buffer = np.empty(
                    list(infer_model.output().shape), dtype=np.uint8
                )

                all_files = []
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    all_files.extend(dir_path.rglob(ext))

                if self.max_images is not None:
                    all_files = all_files[: self.max_images]

                total = len(all_files)
                if total == 0:
                    logger.warning(f"No image files found in directory: {dir_path}")
                    return

                total_time = 0.0
                success_count = 0
                start_time = time.perf_counter()

                for i, image_path in enumerate(all_files):
                    start = time.perf_counter()
                    vec_f32 = self._process_image(image_path, config, output_buffer)
                    if vec_f32 is not None:
                        callback(image_path, vec_f32)
                        success_count += 1
                    total_time += time.perf_counter() - start

                    if (
                        success_count > 0
                        and (success_count % self.profile_batch_size) == 0
                    ):
                        avg_ms = (total_time * 1000) / success_count
                        error_count = (i + 1) - success_count
                        elapsed = time.perf_counter() - start_time
                        logger.info(
                            f"{elapsed:.2f}s elapsed: processed {success_count}/{total}, "
                            f"avg {avg_ms:.2f} ms/image "
                            f"{'(errors: ' + str(error_count) + ')' if error_count > 0 else ''}"
                        )
