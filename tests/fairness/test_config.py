"""Tests for FairnessConfig validation."""

import pytest

from src.fairness.config import FairnessConfig


class TestFairnessConfigDefaults:
    """Tests for FairnessConfig default values."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = FairnessConfig()
        assert config.enabled is False
        assert config.evaluation_dimensions == ["client", "category"]
        assert "jains_index" in config.metrics
        assert "variance" in config.metrics
        assert config.min_samples_per_group == 1
        assert config.max_fpr == 0.05
        assert config.export_detailed_results is True

    def test_disabled_config_skips_validation(self):
        """Test that disabled config does not validate parameters."""
        # This would be invalid if enabled, but should pass when disabled
        config = FairnessConfig(
            enabled=False,
            evaluation_dimensions=["invalid_dimension"],
            metrics=["invalid_metric"],
            min_samples_per_group=-1,
        )
        assert config.enabled is False


class TestFairnessConfigValidation:
    """Tests for FairnessConfig validation when enabled."""

    def test_valid_configuration(self):
        """Test that valid configuration passes validation."""
        config = FairnessConfig(
            enabled=True,
            evaluation_dimensions=["client", "category", "defect_type"],
            metrics=["jains_index", "variance", "performance_gap"],
            min_samples_per_group=5,
            max_fpr=0.1,
        )
        assert config.enabled is True
        assert len(config.evaluation_dimensions) == 3

    def test_invalid_dimension_raises_error(self):
        """Test that invalid dimension raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_dimensions must be from"):
            FairnessConfig(
                enabled=True,
                evaluation_dimensions=["invalid_dimension"],
            )

    def test_invalid_metric_raises_error(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metrics must be from"):
            FairnessConfig(
                enabled=True,
                metrics=["invalid_metric"],
            )

    def test_negative_min_samples_raises_error(self):
        """Test that negative min_samples_per_group raises ValueError."""
        with pytest.raises(ValueError, match="min_samples_per_group must be >= 1"):
            FairnessConfig(
                enabled=True,
                min_samples_per_group=0,
            )

    def test_invalid_max_fpr_raises_error(self):
        """Test that invalid max_fpr raises ValueError."""
        with pytest.raises(ValueError, match="max_fpr must be in"):
            FairnessConfig(
                enabled=True,
                max_fpr=0,
            )

        with pytest.raises(ValueError, match="max_fpr must be in"):
            FairnessConfig(
                enabled=True,
                max_fpr=1.5,
            )

    def test_edge_max_fpr_valid(self):
        """Test that max_fpr=1.0 is valid."""
        config = FairnessConfig(
            enabled=True,
            max_fpr=1.0,
        )
        assert config.max_fpr == 1.0


class TestFairnessConfigValidDimensions:
    """Tests for valid dimensions."""

    def test_client_dimension(self):
        """Test client dimension is valid."""
        config = FairnessConfig(enabled=True, evaluation_dimensions=["client"])
        assert "client" in config.evaluation_dimensions

    def test_category_dimension(self):
        """Test category dimension is valid."""
        config = FairnessConfig(enabled=True, evaluation_dimensions=["category"])
        assert "category" in config.evaluation_dimensions

    def test_defect_type_dimension(self):
        """Test defect_type dimension is valid."""
        config = FairnessConfig(enabled=True, evaluation_dimensions=["defect_type"])
        assert "defect_type" in config.evaluation_dimensions


class TestFairnessConfigValidMetrics:
    """Tests for valid metrics."""

    @pytest.mark.parametrize(
        "metric",
        [
            "jains_index",
            "variance",
            "performance_gap",
            "worst_case",
            "coefficient_of_variation",
            "mean",
            "std",
        ],
    )
    def test_valid_metric(self, metric):
        """Test that each valid metric is accepted."""
        config = FairnessConfig(enabled=True, metrics=[metric])
        assert metric in config.metrics
