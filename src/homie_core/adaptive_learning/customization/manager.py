"""Customization manager — lifecycle management for user-requested customizations."""

import json
import logging
import time
from typing import Optional

from ..storage import LearningStorage
from .generator import CustomizationGenerator

logger = logging.getLogger(__name__)


class CustomizationManager:
    """Manages the lifecycle of user-requested customizations."""

    def __init__(
        self,
        storage: LearningStorage,
        generator: CustomizationGenerator,
    ) -> None:
        self._storage = storage
        self._generator = generator

    def create(self, request: str) -> dict:
        """Create a new customization from a natural language request."""
        # Analyze and generate
        analysis = self._generator.analyze_request(request)
        code = self._generator.generate_code(request, analysis=analysis)

        # Determine file path for generated code
        safe_name = request.lower().replace(" ", "_")[:30].strip("_")
        file_path = f"src/homie_core/adaptive_learning/customization/generated/{safe_name}.py"

        # Apply via evolver (with rollback support)
        version_id = self._generator.apply(file_path, code, reason=f"User request: {request}")

        # Record in history
        self._storage.write_customization(
            request_text=request,
            generated_paths=[file_path],
            version_id=version_id,
            status="active",
        )

        logger.info("Created customization: %s (version: %s)", request[:50], version_id)
        return {
            "status": "active",
            "version_id": version_id,
            "file_path": file_path,
            "request": request,
        }

    def list_customizations(self) -> list[dict]:
        """List all customizations."""
        return self._storage.query_customizations()

    def disable(self, customization_id: int) -> None:
        """Disable a customization."""
        self._storage.update_customization_status(customization_id, "disabled")

    def enable(self, customization_id: int) -> None:
        """Re-enable a disabled customization."""
        self._storage.update_customization_status(customization_id, "active")
