"""Configuration for the cross-instance blob store (host-shared fields only).

Mirrors the shape of :mod:`deerflow.config.memory_config`: this module holds
ONLY the host-shared fields that the factory and call sites read. Backend
private knobs live under ``backend_config`` and are interpreted by the backend
itself, so adding a backend never widens the shared schema.

Why this exists (issue #4189, item 2)
-------------------------------------

Two producers persist blob-shaped data outside the checkpoint payload and
address it with a server-local filesystem path:

1. ``ViewedImageData.actual_path`` (``deerflow/agents/thread_state.py``) --
   written by ``view_image_tool``, read back by ``ViewImageMiddleware``, the
   gateway artifact routes and the IM channels.
2. Oversized tool results externalized by
   ``ToolOutputBudgetMiddleware._externalize`` into the thread's ``outputs``
   tree, addressed by a virtual path that only resolves on the instance that
   holds the thread-data mount.

On a single-gateway deployment both are correct. On a multi-gateway deployment
(Kubernetes behind a load balancer) the instance handling the read is often
not the instance that wrote the file, so the read fails. The blob store
replaces *"where on this machine"* with *"which content"*, so every instance
resolves the same bytes.

The store is **off by default**. Nothing writes to it until a producer is
migrated, and a migrated producer keeps its local path, so turning it on and
off again is behaviour-neutral for existing deployments.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BlobStorageConfig(BaseModel):
    """Host-shared blob storage configuration (backend-agnostic)."""

    enabled: bool = Field(
        default=False,
        description=(
            "Whether producers should write blob content through the blob store. "
            "Defaults to False: the seam exists but nothing writes to it, so a "
            "deployment that does not opt in keeps today's server-local-path "
            "behaviour exactly. A migrated producer keeps its local path "
            "alongside the blob reference, so flipping this off again is also "
            "behaviour-neutral."
        ),
    )
    backend: str = Field(
        default="local_fs",
        description=(
            "Blob store backend selector. Either a registered backend name "
            "(matching a `storage/backends/<name>/` folder that exposes "
            "`STORE_CLASS`, e.g. `local_fs`) or a dotted import path to a "
            "`BlobStore` subclass. The factory resolves this at "
            "`get_blob_store()` time and raises ValueError on failure "
            "(fail-fast: blobs are persistent state, so an unresolved backend "
            "is not silently substituted with a different storage backend)."
        ),
    )
    backend_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backend-private config (a dict), passed verbatim to the backend's "
            "`from_config(backend_config=...)` by the factory. Each backend "
            "self-interprets it. Values live in the host config file "
            "(`config.yaml` `blob_storage.backend_config`); they do not belong "
            "on the shared schema."
        ),
    )


# Global configuration instance.
_blob_storage_config: BlobStorageConfig = BlobStorageConfig()


def get_blob_storage_config() -> BlobStorageConfig:
    """Return the current blob storage configuration.

    ``_blob_storage_config`` is refreshed as a side effect of
    ``get_app_config()`` reloading (via ``_apply_singleton_configs`` ->
    :func:`load_blob_storage_config_from_dict`), mirroring
    :func:`deerflow.config.memory_config.get_memory_config`. If
    ``get_app_config()`` has never been called there is no stale config to
    refresh, so module-level defaults are returned and no config file is
    loaded as a side effect (unit tests rely on that).
    """
    # Lazy import: app_config imports this module, so a top-level import cycles.
    from .app_config import _app_config, get_app_config

    if _app_config is not None:
        try:
            get_app_config()
        except Exception:
            # If the config file is transiently broken (invalid YAML, schema
            # violation, missing env var), keep the last-good singleton so an
            # in-flight turn completes normally instead of crashing.
            logger.warning(
                "Failed to reload app config from get_blob_storage_config(); falling back to cached blob storage config.",
                exc_info=True,
            )
    return _blob_storage_config


def set_blob_storage_config(config: BlobStorageConfig) -> None:
    """Replace the process-wide blob storage configuration (tests / reload)."""
    global _blob_storage_config
    _blob_storage_config = config


def load_blob_storage_config_from_dict(config_dict: dict | None) -> None:
    """Load blob storage configuration from a dictionary.

    Only the three host-shared fields are read. Unknown top-level keys are
    most likely typos, so they are warned about and ignored rather than
    silently accepted into the schema.
    """
    global _blob_storage_config
    config_dict = dict(config_dict or {})
    known = set(BlobStorageConfig.model_fields)
    unknown = sorted(set(config_dict) - known)
    if unknown:
        logger.warning(
            "Ignoring unknown blob_storage config keys (likely typos): %s",
            ", ".join(unknown),
        )
        for key in unknown:
            config_dict.pop(key)
    _blob_storage_config = BlobStorageConfig(**config_dict)
