"""
Centralized configuration for the Visual Search Engine.

This module defines a Config class that holds all settings, reading from
environment variables where available and providing sensible defaults.
"""
import os
from pathlib import Path


class Config:
    """
    Configuration class for the application.
    """

    # --- Qdrant Configuration ---
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")

    # --- Logging Configuration ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Model Configuration ---
    # Default HEF model path for the SimilaritySearchEngine
    DEFAULT_HEF_PATH: Path = (
        Path(__file__).parent / "models" / "resnet_v1_18_feature.hef"
    )
