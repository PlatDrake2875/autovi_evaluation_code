"""Integration tests for FederatedServer with robustness features."""

import numpy as np
import pytest

from src.robustness import (
    RobustnessConfig,
    CoordinateMedianAggregator,
    ZScoreDetector,
    ModelPoisoningAttack,
)
from src.federated.server import FederatedServer


class TestFederatedServerRobustnessInit:
    """Tests for FederatedServer initialization with robustness config."""

    def test_server_without_robustness(self):
        """Test server initializes correctly without robustness."""
        server = FederatedServer(global_bank_size=1000)

        assert server.robustness_config is None
        assert server.robust_aggregator is None
        assert server.client_scorer is None

    def test_server_with_disabled_robustness(self):
        """Test server with disabled robustness config."""
        config = RobustnessConfig(enabled=False)
        server = FederatedServer(global_bank_size=1000, robustness_config=config)

        assert server.robustness_config is not None
        assert server.robust_aggregator is None
        assert server.client_scorer is None

    def test_server_with_coordinate_median(self):
        """Test server initializes CoordinateMedianAggregator when enabled."""
        config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
        )
        server = FederatedServer(global_bank_size=1000, robustness_config=config)

        assert server.robust_aggregator is not None
        assert isinstance(server.robust_aggregator, CoordinateMedianAggregator)
        # num_samples_per_client should be global_bank_size // 10
        assert server.robust_aggregator.num_samples_per_client == 100

    def test_server_with_zscore_detector(self):
        """Test server initializes ZScoreDetector when enabled."""
        config = RobustnessConfig(
            enabled=True,
            client_scoring_method="zscore",
            zscore_threshold=2.5,
        )
        server = FederatedServer(global_bank_size=1000, robustness_config=config)

        assert server.client_scorer is not None
        assert isinstance(server.client_scorer, ZScoreDetector)
        assert server.client_scorer.threshold == 2.5

    def test_server_with_no_client_scoring(self):
        """Test server without client scoring."""
        config = RobustnessConfig(
            enabled=True,
            client_scoring_method="none",
        )
        server = FederatedServer(global_bank_size=1000, robustness_config=config)

        assert server.robust_aggregator is not None
        assert server.client_scorer is None


class TestFederatedServerRobustAggregation:
    """Tests for robust aggregation in FederatedServer."""

    @pytest.fixture
    def robust_server(self):
        """Create a server with robustness enabled."""
        config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
            client_scoring_method="zscore",
            zscore_threshold=3.0,
        )
        return FederatedServer(
            global_bank_size=1000,
            robustness_config=config,
            use_faiss=False,  # Simpler for testing
        )

    @pytest.fixture
    def standard_server(self):
        """Create a server without robustness."""
        return FederatedServer(
            global_bank_size=1000,
            use_faiss=False,
        )

    def test_robust_aggregation_basic(self, robust_server):
        """Test basic robust aggregation flow."""
        feature_dim = 64
        client_coresets = [
            np.random.randn(100, feature_dim) for _ in range(5)
        ]

        robust_server.receive_client_coresets(client_coresets)
        result = robust_server.aggregate()

        assert result is not None
        assert result.shape[1] == feature_dim
        assert robust_server.global_memory_bank is not None

    def test_robust_aggregation_stats(self, robust_server):
        """Test that robust aggregation populates stats correctly."""
        feature_dim = 32
        client_coresets = [
            np.random.randn(100, feature_dim) for _ in range(5)
        ]

        robust_server.receive_client_coresets(client_coresets)
        robust_server.aggregate()

        stats = robust_server.get_stats()

        assert "aggregation_stats" in stats
        agg_stats = stats["aggregation_stats"]
        assert agg_stats.get("robustness_enabled") is True
        assert "aggregation_method" in agg_stats
        assert agg_stats["aggregation_method"] == "coordinate_median"

    def test_client_scoring_in_stats(self, robust_server):
        """Test that client scores are recorded in stats."""
        feature_dim = 32
        # Create normal clients
        client_coresets = [
            np.random.randn(100, feature_dim) for _ in range(4)
        ]
        # Add one outlier client
        client_coresets.append(np.random.randn(100, feature_dim) * 1000)

        robust_server.receive_client_coresets(client_coresets)
        robust_server.aggregate()

        stats = robust_server.get_stats()
        agg_stats = stats["aggregation_stats"]

        assert "client_scores" in agg_stats
        assert len(agg_stats["client_scores"]) == 5
        assert "num_outliers" in agg_stats
        assert "outlier_indices" in agg_stats

    def test_standard_aggregation_fallback(self, standard_server):
        """Test that server falls back to standard aggregation when no robustness."""
        feature_dim = 32
        client_coresets = [
            np.random.randn(100, feature_dim) for _ in range(3)
        ]

        standard_server.receive_client_coresets(client_coresets)
        result = standard_server.aggregate()

        assert result is not None
        stats = standard_server.get_stats()
        assert stats["aggregation_stats"].get("robustness_enabled") is not True


class TestRobustnessUnderAttack:
    """Tests for robustness under attack scenarios."""

    def test_robust_vs_baseline_under_attack(self):
        """Compare robust vs baseline aggregation under attack."""
        feature_dim = 32
        num_honest = 5
        num_malicious = 2

        # Create honest client updates (values around 1.0)
        rng = np.random.default_rng(42)
        honest_updates = [
            rng.normal(1.0, 0.1, size=(100, feature_dim))
            for _ in range(num_honest)
        ]

        # Create attack
        attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=100.0, seed=42)

        # Full client list with malicious clients
        all_updates = honest_updates + [
            rng.normal(1.0, 0.1, size=(100, feature_dim))
            for _ in range(num_malicious)
        ]
        malicious_indices = list(range(num_honest, num_honest + num_malicious))
        attacked_updates = attack.apply(all_updates, malicious_indices)

        # Robust server
        robust_config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
        )
        robust_server = FederatedServer(
            global_bank_size=100,
            robustness_config=robust_config,
            use_faiss=False,
        )
        robust_server.receive_client_coresets(attacked_updates)
        robust_result = robust_server.aggregate()

        # Check that robust result is closer to honest value (1.0)
        robust_mean = np.mean(robust_result)

        # The median should resist the scaled attack
        # With 2/7 malicious, median should still be close to 1.0
        assert abs(robust_mean - 1.0) < 10  # Should be much closer to 1 than 100

    def test_outlier_detection_under_attack(self):
        """Test that attacked clients are detected as outliers."""
        feature_dim = 32
        num_honest = 8
        num_malicious = 2

        rng = np.random.default_rng(42)

        # Create honest client updates
        honest_updates = [
            rng.normal(1.0, 0.1, size=(100, feature_dim))
            for _ in range(num_honest)
        ]

        # Create malicious updates (extreme values)
        malicious_updates = [
            rng.normal(1000.0, 1.0, size=(100, feature_dim))
            for _ in range(num_malicious)
        ]

        all_updates = honest_updates + malicious_updates

        # Server with client scoring
        config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
            client_scoring_method="zscore",
            zscore_threshold=2.0,  # Lower threshold to detect outliers
        )
        server = FederatedServer(
            global_bank_size=100,
            robustness_config=config,
            use_faiss=False,
        )

        server.receive_client_coresets(all_updates)
        server.aggregate()

        stats = server.get_stats()
        agg_stats = stats["aggregation_stats"]

        # Should detect the malicious clients as outliers
        assert agg_stats["num_outliers"] >= 1
        # The malicious clients are at indices 8 and 9
        detected_outliers = set(agg_stats["outlier_indices"])
        malicious_set = {8, 9}
        # At least one malicious client should be detected
        assert len(detected_outliers & malicious_set) >= 1


class TestEndToEndRobustness:
    """End-to-end tests for the complete robustness pipeline."""

    def test_full_pipeline_with_attack_recovery(self):
        """Test full pipeline: attack -> robust aggregation -> reasonable output."""
        feature_dim = 64
        num_clients = 10
        num_malicious = 3  # 30% malicious

        rng = np.random.default_rng(42)

        # Generate honest client data (embeddings around 0)
        client_updates = [
            rng.normal(0, 1, size=(100, feature_dim))
            for _ in range(num_clients)
        ]

        # Apply attack to some clients
        attack = ModelPoisoningAttack(attack_type="sign_flip", seed=42)
        malicious_indices = list(range(num_malicious))
        attacked_updates = attack.apply(client_updates, malicious_indices)

        # Create robust server
        config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
            client_scoring_method="zscore",
        )
        server = FederatedServer(
            global_bank_size=100,
            robustness_config=config,
            use_faiss=False,
        )

        # Run aggregation
        server.receive_client_coresets(attacked_updates)
        result = server.aggregate()

        # Verify output is reasonable (not dominated by attacked values)
        assert result is not None
        assert result.shape[1] == feature_dim

        # The median should be close to 0 (honest value), not dominated by flipped values
        # With 30% malicious, median should still reflect honest majority
        mean_result = np.mean(np.abs(result))
        # Should be within reasonable range of standard normal
        assert mean_result < 5.0  # Not dominated by extreme values

    def test_aggregation_reproducibility(self):
        """Test that robust aggregation is reproducible with same seed."""
        feature_dim = 32
        client_updates = [np.random.randn(100, feature_dim) for _ in range(5)]

        config = RobustnessConfig(
            enabled=True,
            aggregation_method="coordinate_median",
        )

        # First run
        server1 = FederatedServer(
            global_bank_size=100,
            robustness_config=config,
            use_faiss=False,
        )
        server1.receive_client_coresets(client_updates)
        result1 = server1.aggregate(seed=123)

        # Second run with same seed
        server2 = FederatedServer(
            global_bank_size=100,
            robustness_config=config,
            use_faiss=False,
        )
        server2.receive_client_coresets(client_updates)
        result2 = server2.aggregate(seed=123)

        np.testing.assert_array_equal(result1, result2)
