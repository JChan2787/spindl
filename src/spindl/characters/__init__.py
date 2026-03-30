"""
Character Card system for spindl.

Provides ST V2-compatible character card loading and management,
replacing the legacy PersonaLoader with full ecosystem interoperability.
"""

from .exporter import BatchExportResult, CharacterExporter, ExportError
from .importer import (
    CharacterImporter,
    ImportError,
    ImportResult,
    ValidationResult,
    embed_chara_in_png,
    extract_chara_from_png,
)
from .loader import CharacterLoader
from .models import (
    CharacterBook,
    CharacterBookEntry,
    CharacterCard,
    CharacterCardData,
    GenerationConfig,
    SpindlExtensions,
)

__all__ = [
    # Loader
    "CharacterLoader",
    # Importer
    "CharacterImporter",
    "ImportResult",
    "ValidationResult",
    "ImportError",
    # PNG utilities
    "extract_chara_from_png",
    "embed_chara_in_png",
    # Exporter
    "CharacterExporter",
    "BatchExportResult",
    "ExportError",
    # Models
    "CharacterCard",
    "CharacterCardData",
    "CharacterBook",
    "CharacterBookEntry",
    "SpindlExtensions",
    "GenerationConfig",
]
