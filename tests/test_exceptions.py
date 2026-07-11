"""Tests for the typed exception hierarchy."""

from crowdwisdom_quant.utils.exceptions import (
    ApiKeyMissingError,
    ApiRateLimitError,
    ApiResponseError,
    ColumnMappingError,
    ConfigurationError,
    CrowdWisdomError,
    DataError,
    DataIntegrityError,
    DataNotFoundError,
    DatabaseError,
    FeatureEngineeringError,
    HyperparameterError,
    LeakageDetectedError,
    MergeError,
    ModelError,
    ModelNotTrainedError,
    PreprocessingError,
    ReportError,
    ScraperError,
    ValidationError,
)


class TestExceptionHierarchy:
    """All exceptions should inherit from CrowdWisdomError.

    This lets callers catch ``CrowdWisdomError`` to handle any
    application-level failure without catching system exceptions.
    """

    def test_base_class(self) -> None:
        assert issubclass(CrowdWisdomError, Exception)

    def test_data_exceptions(self) -> None:
        assert issubclass(DataError, CrowdWisdomError)
        assert issubclass(DatabaseError, DataError)
        assert issubclass(DataNotFoundError, DataError)
        assert issubclass(DataIntegrityError, DataError)

    def test_scraper_exceptions(self) -> None:
        assert issubclass(ScraperError, CrowdWisdomError)
        assert issubclass(ApiKeyMissingError, ScraperError)
        assert issubclass(ApiRateLimitError, ScraperError)
        assert issubclass(ApiResponseError, ScraperError)

    def test_preprocessing_exceptions(self) -> None:
        assert issubclass(PreprocessingError, CrowdWisdomError)
        assert issubclass(MergeError, PreprocessingError)
        assert issubclass(ColumnMappingError, PreprocessingError)

    def test_feature_exceptions(self) -> None:
        assert issubclass(FeatureEngineeringError, CrowdWisdomError)
        assert issubclass(LeakageDetectedError, FeatureEngineeringError)

    def test_model_exceptions(self) -> None:
        assert issubclass(ModelError, CrowdWisdomError)
        assert issubclass(ModelNotTrainedError, ModelError)
        assert issubclass(HyperparameterError, ModelError)
        assert issubclass(ValidationError, ModelError)

    def test_config_exception(self) -> None:
        assert issubclass(ConfigurationError, CrowdWisdomError)

    def test_report_exception(self) -> None:
        assert issubclass(ReportError, CrowdWisdomError)

    def test_exceptions_carry_message(self) -> None:
        msg = "test error message"
        for exc_cls in [
            CrowdWisdomError,
            DatabaseError,
            ApiKeyMissingError,
            PreprocessingError,
            ModelError,
        ]:
            exc = exc_cls(msg)
            assert str(exc) == msg, f"{exc_cls.__name__} did not carry message"
