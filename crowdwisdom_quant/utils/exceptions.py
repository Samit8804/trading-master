"""Typed exception hierarchy for CrowdWisdomTrading.

Categorises all runtime errors so callers (and the CLI) can handle
specific failure modes without parsing string messages.
"""


class CrowdWisdomError(Exception):
    """Base for all application-level errors."""


# ── Data layer ─────────────────────────────────────────────────────────────

class DataError(CrowdWisdomError):
    """Generic data-access error."""


class DatabaseError(DataError):
    """Database connection, query, or constraint failure."""


class DataNotFoundError(DataError):
    """Required file, table, or record does not exist."""


class DataIntegrityError(DataError):
    """Schema violation, duplicate key, or corrupted data."""


# ── Scraper ─────────────────────────────────────────────────────────────────

class ScraperError(CrowdWisdomError):
    """Generic scraper error."""


class ApiKeyMissingError(ScraperError):
    """No API key configured for an external data source."""


class ApiRateLimitError(ScraperError):
    """External API rate limit exceeded."""


class ApiResponseError(ScraperError):
    """External API returned an unexpected response."""


# ── Preprocessing ──────────────────────────────────────────────────────────

class PreprocessingError(CrowdWisdomError):
    """Data cleaning / transformation failure."""


class MergeError(PreprocessingError):
    """Trade-event merge failure (e.g. incompatible schemas)."""


class ColumnMappingError(PreprocessingError):
    """Missing or unexpected columns in input data."""


# ── Feature engineering ─────────────────────────────────────────────────────

class FeatureEngineeringError(CrowdWisdomError):
    """Feature computation failure."""


class LeakageDetectedError(FeatureEngineeringError):
    """Future data detected in features — possible look-ahead bias."""


# ── Model ───────────────────────────────────────────────────────────────────

class ModelError(CrowdWisdomError):
    """Model training / prediction failure."""


class ModelNotTrainedError(ModelError):
    """Attempted prediction without a trained model."""


class HyperparameterError(ModelError):
    """Invalid or incompatible hyperparameter configuration."""


class ValidationError(ModelError):
    """Walk-forward validation produced no valid folds."""


# ── Configuration ──────────────────────────────────────────────────────────

class ConfigurationError(CrowdWisdomError):
    """Invalid or missing configuration."""


# ── Reporting ───────────────────────────────────────────────────────────────

class ReportError(CrowdWisdomError):
    """Report generation failure."""
