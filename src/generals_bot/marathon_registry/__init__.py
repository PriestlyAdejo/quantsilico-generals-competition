"""Minimum canonical Marathon registry (EXECUTION_PLAN Stage 3)."""

from generals_bot.marathon_registry.registry import (
    KINDS,
    PPO_SEMANTICS,
    SCHEMA_VERSION,
    Registry,
    RegistryError,
    canonical_id,
)

__all__ = [
    "KINDS",
    "PPO_SEMANTICS",
    "SCHEMA_VERSION",
    "Registry",
    "RegistryError",
    "canonical_id",
]
