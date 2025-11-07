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

    - Uses context-managed VDevice/config setup for safe cleanup.
    - Supports folder or single-image inference.
    - Progress + timing information.
    - Calls a user-supplied callback with (rel_path, vec_f32).
    """

    def __init__(
        self,
        hef_path: Path,
        base_folder: Path,
        image_size=(224, 224),
        profile_batch_size: int = 100,
        max_images: Optional[int] = 5000,
    ):
        self.hef_path = Path(hef_path)
        self.base_folder = Path(base_folder)
        self.image_size = image_size
        self.profile_batch_size = profile_batch_size
        self.max_images = max_images

    # ---------------------------------------------------------------------
    def _process_image(
        self, rel_path: Path, config, bindings, output_buffer
    ) -> Optional[np.ndarray]:
        """Run inference on a single image and return normalized embedding."""
        image_path = self.base_folder / rel_path
        try:
            # Load and resize
            with Image.open(image_path) as img:
                img = img.convert("RGB").resize(self.image_size, Image.LANCZOS)
                input_buffer = np.asarray(img, dtype=np.uint8)

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
                logger.warning(f"Zero norm vector for {rel_path}")
                return None
            return vec_f32

        except Exception as e:
            logger.warning(f"[ERROR] Failed to process {rel_path}: {e}")
            return None

    # ---------------------------------------------------------------------
    def _process_image_dir(
        self,
        relative_path: Path,
        config,
        bindings,
        output_buffer,
        callback: Callable[[Path, np.ndarray], None],
    ):
        """Traverse a folder or process a single file."""
        path = self.base_folder / relative_path

        if not path.exists():
            logger.warning(f"Path not found: {path}")
            return

        total_time = 0.0
        success_count = 0

        if path.is_dir():
            all_files = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                all_files.extend(path.rglob(ext))

            # Limit to max_images if specified
            if self.max_images is not None:
                all_files = all_files[: self.max_images]

            total = len(all_files)
            start_time = time.perf_counter()

            for i, image_path in enumerate(all_files):
                rel_path = image_path.resolve().relative_to(self.base_folder)
                start = time.perf_counter()

                vec_f32 = self._process_image(rel_path, config, bindings, output_buffer)
                if vec_f32 is not None:
                    callback(rel_path, vec_f32)
                    success_count += 1
                total_time += time.perf_counter() - start

                if (success_count % self.profile_batch_size) == 0:
                    avg_ms = (total_time * 1000) / success_count
                    error_count = (i + 1) - success_count
                    elapsed = time.perf_counter() - start_time
                    logger.info(
                        f"{elapsed:.2f}s elapsed: processed {success_count}/{total}, "
                        f"avg {avg_ms:.2f} ms/image "
                        f"{'(errors: ' + str(error_count) + ')' if error_count > 0 else ''}"
                    )

        else:
            logger.info(f"Processing single image: {relative_path}")
            vec_f32 = self._process_image(
                relative_path, config, bindings, output_buffer
            )
            if vec_f32 is not None:
                callback(relative_path, vec_f32)

    # ---------------------------------------------------------------------
    def process(
        self, relative_path: Path, callback: Callable[[Path, np.ndarray], None]
    ):
        """
        Main entry point.
        Processes all images under `relative_path` (file or folder)
        and sends (rel_path, vec_f32) to the callback.
        """
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
                self._process_image_dir(
                    relative_path, config, bindings, output_buffer, callback
                )
