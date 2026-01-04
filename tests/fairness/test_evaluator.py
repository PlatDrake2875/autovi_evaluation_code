"""Tests for FairnessEvaluator."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.fairness.config import FairnessConfig
from src.fairness.evaluator import (
    FairnessEvaluator,
    GroupEvaluationResult,
    FairnessEvaluationResult,
)


@dataclass
class MockAnomalyMap:
    """Mock AnomalyMap for testing."""

    file_path: str
    np_array: np.ndarray = None

    def __post_init__(self):
        if self.np_array is None:
            self.np_array = np.random.rand(100, 100)


class MockThresholdMetrics:
    """Mock ThresholdMetrics for testing."""

    def __init__(self, anomaly_maps, gt_maps=None):
        self.anomaly_maps = anomaly_maps
        self.gt_maps = gt_maps or [None] * len(anomaly_maps)
        self.anomaly_thresholds = np.linspace(1.0, 0.0, 100)

    def reduce_to_images(self, take_anomaly_maps):
        """Return a new MockThresholdMetrics with filtered maps."""
        indices = [self.anomaly_maps.index(m) for m in take_anomaly_maps]
        return MockThresholdMetrics(
            anomaly_maps=take_anomaly_maps,
            gt_maps=[self.gt_maps[i] for i in indices],
        )

    def get_fp_rates(self):
        """Return mock FP rates."""
        return np.linspace(0.0, 1.0, 100)

    def get_mean_spros(self):
        """Return mock mean sPROs."""
        return np.linspace(0.0, 1.0, 100)


class TestFairnessEvaluatorInit:
    """Tests for FairnessEvaluator initialization."""

    def test_default_client_mapping(self):
        """Test that default client mapping is used."""
        config = FairnessConfig(enabled=True)
        evaluator = FairnessEvaluator(config)

        assert evaluator.client_mapping is not None
        assert 0 in evaluator.client_mapping
        assert "engine_wiring" in evaluator.client_mapping[0]

    def test_custom_client_mapping(self):
        """Test that custom client mapping is used."""
        config = FairnessConfig(enabled=True)
        custom_mapping = {0: ["cat_a"], 1: ["cat_b"]}
        evaluator = FairnessEvaluator(config, client_mapping=custom_mapping)

        assert evaluator.client_mapping == custom_mapping
        assert evaluator._category_to_client["cat_a"] == 0
        assert evaluator._category_to_client["cat_b"] == 1

    def test_category_to_client_reverse_mapping(self):
        """Test that category to client reverse mapping is built correctly."""
        config = FairnessConfig(enabled=True)
        custom_mapping = {0: ["cat_a", "cat_b"], 1: ["cat_c"]}
        evaluator = FairnessEvaluator(config, client_mapping=custom_mapping)

        assert evaluator._category_to_client["cat_a"] == 0
        assert evaluator._category_to_client["cat_b"] == 0
        assert evaluator._category_to_client["cat_c"] == 1


class TestGroupAnomMapsByCategory:
    """Tests for grouping anomaly maps by category."""

    def test_group_by_category(self):
        """Test grouping anomaly maps by category."""
        config = FairnessConfig(enabled=True)
        evaluator = FairnessEvaluator(config)

        # Create mock anomaly maps with different categories
        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/dent/img2.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img3.png"),
        ]

        grouped = evaluator._group_anomaly_maps_by_category(anomaly_maps)

        assert "engine_wiring" in grouped
        assert "tank_screw" in grouped
        assert len(grouped["engine_wiring"]) == 2
        assert len(grouped["tank_screw"]) == 1


class TestGroupAnomMapsByDefectType:
    """Tests for grouping anomaly maps by defect type."""

    def test_group_by_defect_type(self):
        """Test grouping anomaly maps by defect type."""
        config = FairnessConfig(enabled=True)
        evaluator = FairnessEvaluator(config)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/crack/img2.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img3.png"),
            MockAnomalyMap("/output/tank_screw/good/img4.png"),
        ]

        grouped = evaluator._group_anomaly_maps_by_defect_type(anomaly_maps)

        assert "crack" in grouped
        assert "scratch" in grouped
        assert "good" in grouped
        assert len(grouped["crack"]) == 2
        assert len(grouped["scratch"]) == 1
        assert len(grouped["good"]) == 1


class TestGroupAnomMapsByClient:
    """Tests for grouping anomaly maps by client."""

    def test_group_by_client(self):
        """Test grouping anomaly maps by client."""
        config = FairnessConfig(enabled=True)
        custom_mapping = {
            0: ["engine_wiring"],
            1: ["tank_screw"],
        }
        evaluator = FairnessEvaluator(config, client_mapping=custom_mapping)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/dent/img2.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img3.png"),
        ]

        grouped = evaluator._group_anomaly_maps_by_client(anomaly_maps)

        assert 0 in grouped
        assert 1 in grouped
        assert len(grouped[0]) == 2
        assert len(grouped[1]) == 1


class TestEvaluateGroup:
    """Tests for evaluating a single group."""

    def test_evaluate_group_returns_result(self):
        """Test that _evaluate_group returns GroupEvaluationResult."""
        config = FairnessConfig(enabled=True)
        evaluator = FairnessEvaluator(config)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/dent/img2.png"),
        ]
        metrics = MockThresholdMetrics(anomaly_maps)

        result = evaluator._evaluate_group(metrics, anomaly_maps, "test_group")

        assert isinstance(result, GroupEvaluationResult)
        assert result.group_name == "test_group"
        assert result.num_images == 2
        assert isinstance(result.auc_spro, float)


class TestEvaluateByCategory:
    """Tests for category-level fairness evaluation."""

    def test_evaluate_by_category(self):
        """Test evaluating fairness by category."""
        config = FairnessConfig(enabled=True, min_samples_per_group=1)
        evaluator = FairnessEvaluator(config)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/dent/img2.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img3.png"),
        ]
        metrics = MockThresholdMetrics(anomaly_maps)

        result = evaluator.evaluate_by_category(metrics)

        assert isinstance(result, FairnessEvaluationResult)
        assert result.dimension == "category"
        assert "engine_wiring" in result.group_results
        assert "tank_screw" in result.group_results
        assert result.fairness_metrics.n_groups == 2

    def test_min_samples_filtering(self):
        """Test that groups with fewer samples are skipped."""
        config = FairnessConfig(enabled=True, min_samples_per_group=3)
        evaluator = FairnessEvaluator(config)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/dent/img2.png"),
            MockAnomalyMap("/output/engine_wiring/good/img3.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img4.png"),  # Only 1 sample
        ]
        metrics = MockThresholdMetrics(anomaly_maps)

        result = evaluator.evaluate_by_category(metrics)

        # tank_screw should be skipped (only 1 sample < min 3)
        assert "engine_wiring" in result.group_results
        assert "tank_screw" not in result.group_results
        assert result.fairness_metrics.n_groups == 1


class TestEvaluateByClient:
    """Tests for client-level fairness evaluation."""

    def test_evaluate_by_client(self):
        """Test evaluating fairness by client."""
        config = FairnessConfig(enabled=True, min_samples_per_group=1)
        custom_mapping = {
            0: ["engine_wiring"],
            1: ["tank_screw"],
        }
        evaluator = FairnessEvaluator(config, client_mapping=custom_mapping)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/dent/img2.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img3.png"),
        ]
        metrics = MockThresholdMetrics(anomaly_maps)

        result = evaluator.evaluate_by_client(metrics)

        assert isinstance(result, FairnessEvaluationResult)
        assert result.dimension == "client"
        assert "client_0" in result.group_results
        assert "client_1" in result.group_results


class TestEvaluateByDefectType:
    """Tests for defect-type-level fairness evaluation."""

    def test_evaluate_by_defect_type(self):
        """Test evaluating fairness by defect type."""
        config = FairnessConfig(enabled=True, min_samples_per_group=1)
        evaluator = FairnessEvaluator(config)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/engine_wiring/crack/img2.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img3.png"),
            MockAnomalyMap("/output/tank_screw/good/img4.png"),
        ]
        metrics = MockThresholdMetrics(anomaly_maps)

        result = evaluator.evaluate_by_defect_type(metrics)

        assert isinstance(result, FairnessEvaluationResult)
        assert result.dimension == "defect_type"
        assert "crack" in result.group_results
        assert "scratch" in result.group_results
        # "good" should be skipped
        assert "good" not in result.group_results


class TestEvaluateAllDimensions:
    """Tests for evaluating all configured dimensions."""

    def test_evaluate_all_dimensions(self):
        """Test evaluating all configured dimensions."""
        config = FairnessConfig(
            enabled=True,
            evaluation_dimensions=["client", "category"],
            min_samples_per_group=1,
        )
        custom_mapping = {
            0: ["engine_wiring"],
            1: ["tank_screw"],
        }
        evaluator = FairnessEvaluator(config, client_mapping=custom_mapping)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img2.png"),
        ]
        metrics = MockThresholdMetrics(anomaly_maps)

        results = evaluator.evaluate_all_dimensions(metrics)

        assert "client" in results
        assert "category" in results
        assert results["client"].dimension == "client"
        assert results["category"].dimension == "category"


class TestFairnessMetricsComputation:
    """Tests for fairness metrics computation in evaluator."""

    def test_fairness_metrics_computed(self):
        """Test that fairness metrics are computed correctly."""
        config = FairnessConfig(enabled=True, min_samples_per_group=1)
        evaluator = FairnessEvaluator(config)

        anomaly_maps = [
            MockAnomalyMap("/output/engine_wiring/crack/img1.png"),
            MockAnomalyMap("/output/tank_screw/scratch/img2.png"),
        ]
        metrics = MockThresholdMetrics(anomaly_maps)

        result = evaluator.evaluate_by_category(metrics)

        # Check that fairness metrics are populated
        fm = result.fairness_metrics
        assert fm.n_groups == 2
        assert 0 < fm.jains_index <= 1.0
        assert fm.variance >= 0
        assert fm.performance_gap >= 0
