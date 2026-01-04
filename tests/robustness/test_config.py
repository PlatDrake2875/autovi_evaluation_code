"""Tests for RobustnessConfig class."""

import pytest

from src.robustness.config import RobustnessConfig


class TestRobustnessConfigDefaults:
    """Tests for RobustnessConfig default values."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RobustnessConfig()
        assert config.enabled is False
        assert config.aggregation_method == "coordinate_median"
        assert config.num_byzantine == 0
        assert config.trim_fraction == 0.1
        assert config.client_scoring_method == "zscore"
        assert config.zscore_threshold == 3.0

    def test_disabled_config_allows_any_values(self):
        """Test that disabled config skips validation."""
        # Invalid values should be allowed when disabled
        config = RobustnessConfig(
            enabled=False,
            aggregation_method="invalid_method",
            num_byzantine=-5,
            trim_fraction=0.99,
            zscore_threshold=-1.0,
        )
        assert config.enabled is False
        assert config.aggregation_method == "invalid_method"


class TestRobustnessConfigValidation:
    """Tests for RobustnessConfig validation when enabled."""

    def test_valid_coordinate_median_config(self):
        """Test valid configuration with coordinate_median."""
        config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
            num_byzantine=2,
            trim_fraction=0.2,
            client_scoring_method="zscore",
            zscore_threshold=3.0,
        )
        assert config.enabled is True
        assert config.aggregation_method == "coordinate_median"

    def test_valid_trimmed_mean_config(self):
        """Test valid configuration with trimmed_mean."""
        config = RobustnessConfig(
            enabled=True,
            aggregation_method="trimmed_mean",
            trim_fraction=0.25,
        )
        assert config.aggregation_method == "trimmed_mean"

    def test_valid_no_client_scoring(self):
        """Test valid configuration with no client scoring."""
        config = RobustnessConfig(
            enabled=True,
            client_scoring_method="none",
        )
        assert config.client_scoring_method == "none"


class TestAggregationMethodValidation:
    """Tests for aggregation method validation."""

    def test_invalid_aggregation_method(self):
        """Test error when aggregation method is invalid."""
        with pytest.raises(ValueError, match="aggregation_method must be one of"):
            RobustnessConfig(enabled=True, aggregation_method="invalid")

    def test_aggregation_method_case_sensitive(self):
        """Test that aggregation method is case sensitive."""
        with pytest.raises(ValueError, match="aggregation_method must be one of"):
            RobustnessConfig(enabled=True, aggregation_method="COORDINATE_MEDIAN")


class TestNumByzantineValidation:
    """Tests for num_byzantine validation."""

    def test_negative_num_byzantine(self):
        """Test error when num_byzantine is negative."""
        with pytest.raises(ValueError, match="num_byzantine must be non-negative"):
            RobustnessConfig(enabled=True, num_byzantine=-1)

    def test_zero_num_byzantine_valid(self):
        """Test that zero is valid for num_byzantine."""
        config = RobustnessConfig(enabled=True, num_byzantine=0)
        assert config.num_byzantine == 0

    def test_positive_num_byzantine_valid(self):
        """Test that positive values are valid for num_byzantine."""
        config = RobustnessConfig(enabled=True, num_byzantine=5)
        assert config.num_byzantine == 5


class TestTrimFractionValidation:
    """Tests for trim_fraction validation."""

    def test_trim_fraction_zero(self):
        """Test error when trim_fraction is zero."""
        with pytest.raises(ValueError, match="trim_fraction must be in"):
            RobustnessConfig(enabled=True, trim_fraction=0.0)

    def test_trim_fraction_at_half(self):
        """Test error when trim_fraction is 0.5."""
        with pytest.raises(ValueError, match="trim_fraction must be in"):
            RobustnessConfig(enabled=True, trim_fraction=0.5)

    def test_trim_fraction_above_half(self):
        """Test error when trim_fraction exceeds 0.5."""
        with pytest.raises(ValueError, match="trim_fraction must be in"):
            RobustnessConfig(enabled=True, trim_fraction=0.6)

    def test_trim_fraction_negative(self):
        """Test error when trim_fraction is negative."""
        with pytest.raises(ValueError, match="trim_fraction must be in"):
            RobustnessConfig(enabled=True, trim_fraction=-0.1)

    def test_valid_trim_fraction_boundaries(self):
        """Test valid trim_fraction near boundaries."""
        config_low = RobustnessConfig(enabled=True, trim_fraction=0.01)
        assert config_low.trim_fraction == 0.01

        config_high = RobustnessConfig(enabled=True, trim_fraction=0.49)
        assert config_high.trim_fraction == 0.49


class TestClientScoringMethodValidation:
    """Tests for client_scoring_method validation."""

    def test_invalid_client_scoring_method(self):
        """Test error when client_scoring_method is invalid."""
        with pytest.raises(ValueError, match="client_scoring_method must be one of"):
            RobustnessConfig(enabled=True, client_scoring_method="invalid")

    def test_client_scoring_method_case_sensitive(self):
        """Test that client_scoring_method is case sensitive."""
        with pytest.raises(ValueError, match="client_scoring_method must be one of"):
            RobustnessConfig(enabled=True, client_scoring_method="ZSCORE")


class TestZscoreThresholdValidation:
    """Tests for zscore_threshold validation."""

    def test_zscore_threshold_zero(self):
        """Test error when zscore_threshold is zero."""
        with pytest.raises(ValueError, match="zscore_threshold must be positive"):
            RobustnessConfig(enabled=True, zscore_threshold=0.0)

    def test_zscore_threshold_negative(self):
        """Test error when zscore_threshold is negative."""
        with pytest.raises(ValueError, match="zscore_threshold must be positive"):
            RobustnessConfig(enabled=True, zscore_threshold=-1.0)

    def test_zscore_threshold_positive(self):
        """Test valid positive zscore_threshold values."""
        config = RobustnessConfig(enabled=True, zscore_threshold=2.5)
        assert config.zscore_threshold == 2.5

        config_small = RobustnessConfig(enabled=True, zscore_threshold=0.1)
        assert config_small.zscore_threshold == 0.1
