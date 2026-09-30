"""Expected failures that the CLI can report without a traceback."""


class CadError(Exception):
    """An unreadable, unsupported, or unusable CAD model."""


class DependencyError(Exception):
    """The native pythonOCC dependency cannot be imported."""


class DatabaseError(Exception):
    """The index cannot be opened with this schema version."""


class DescriptorError(CadError):
    """Descriptor generation, compatibility, or persistence failed."""


class DetailedSimilarityError(CadError):
    """Detailed geometry preprocessing, cache, or reranking failed."""
