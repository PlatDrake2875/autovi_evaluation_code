"""Tests for client scoring and anomaly detection."""

import numpy as np
import pytest

from src.robustness import ClientScore, ZScoreDetector


class TestClientScore:
    """Tests for ClientScore dataclass."""

    def test_default_details(self):
        """Test that details defaults to empty dict."""
        score = ClientScore(client_id=0, score=1.5, is_outlier=False)
        assert score.details == {}

    def test_with_details(self):
        """Test ClientScore with custom details."""
        details = {"mean_norm": 1.0, "z_score": 2.5}
        score = ClientScore(client_id=1, score=2.5, is_outlier=True, details=details)
        assert score.details == details
        assert score.is_outlier == True


class TestZScoreDetector:
    """Tests for ZScoreDetector."""

    def test_normal_clients_not_flagged(self):
        """Test that normal clients are not flagged as outliers."""
        detector = ZScoreDetector(threshold=3.0)

        # Create 5 similar clients
        rng = np.random.default_rng(42)
        client_updates = [
            rng.normal(1.0, 0.1, size=(100, 32))
            for _ in range(5)
        ]

        scores = detector.score_clients(client_updates)

        assert len(scores) == 5
        assert all(not s.is_outlier for s in scores)
        assert all(s.score < 3.0 for s in scores)

    def test_outlier_detected(self):
        """Test that an outlier client is detected."""
        detector = ZScoreDetector(threshold=1.5)

        # Create 10 normal clients (more clients = higher Z-score for outlier)
        rng = np.random.default_rng(42)
        client_updates = [
            rng.normal(1.0, 0.1, size=(100, 32))
            for _ in range(10)
        ]

        # Add 1 extreme outlier with much larger norms
        client_updates.append(rng.normal(1000.0, 1.0, size=(100, 32)))

        scores = detector.score_clients(client_updates)

        assert len(scores) == 11
        # The last client should be flagged as outlier
        assert scores[10].is_outlier == True
        assert scores[10].score > 1.5
        # Normal clients should not be flagged
        assert all(not s.is_outlier for s in scores[:10])

    def test_threshold_parameter(self):
        """Test that threshold parameter affects detection."""
        # Create clients with one moderate outlier
        rng = np.random.default_rng(42)
        client_updates = [
            rng.normal(1.0, 0.1, size=(100, 32))
            for _ in range(4)
        ]
        # Add a more extreme outlier
        client_updates.append(rng.normal(50.0, 0.1, size=(100, 32)))

        # Low threshold should detect it
        detector_low = ZScoreDetector(threshold=1.0)
        scores_low = detector_low.score_clients(client_updates)
        assert scores_low[4].is_outlier == True

        # Very high threshold should not detect it
        detector_high = ZScoreDetector(threshold=10.0)
        scores_high = detector_high.score_clients(client_updates)
        assert scores_high[4].is_outlier == False

    def test_empty_input(self):
        """Test with empty input."""
        detector = ZScoreDetector()
        scores = detector.score_clients([])
        assert scores == []

    def test_single_client(self):
        """Test with single client (no variance possible)."""
        detector = ZScoreDetector()
        client_updates = [np.random.randn(100, 32)]

        scores = detector.score_clients(client_updates)

        assert len(scores) == 1
        # Single client can't be an outlier (no variance)
        assert scores[0].score == 0.0
        assert scores[0].is_outlier == False

    def test_invalid_threshold(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError, match="Threshold must be positive"):
            ZScoreDetector(threshold=0)

        with pytest.raises(ValueError, match="Threshold must be positive"):
            ZScoreDetector(threshold=-1.0)

    def test_score_details(self):
        """Test that score details contain expected fields."""
        detector = ZScoreDetector()
        client_updates = [np.random.randn(100, 32) for _ in range(3)]

        scores = detector.score_clients(client_updates)

        for score in scores:
            assert "mean_norm" in score.details
            assert "std_norm" in score.details
            assert "max_norm" in score.details
            assert "z_mean_norm" in score.details
            assert "z_std_norm" in score.details
            assert "z_max_norm" in score.details

    def test_repr(self):
        """Test string representation."""
        detector = ZScoreDetector(threshold=2.5)
        assert "ZScoreDetector" in repr(detector)
        assert "2.5" in repr(detector)
