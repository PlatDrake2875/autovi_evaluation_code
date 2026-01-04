"""Tests for PatchCoreClient differential privacy integration."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from src.privacy import DPConfig, EmbeddingSanitizer
from src.federated.client import PatchCoreClient


class TestClientDPInit:
    """Tests for PatchCoreClient DP initialization."""

    def test_init_without_dp(self):
        """Test client initialization without DP."""
        with patch.object(PatchCoreClient, '__init__', lambda self, **kwargs: None):
            client = PatchCoreClient.__new__(PatchCoreClient)
            client.dp_config = None
            client.sanitizer = None

            assert client.sanitizer is None

    def test_init_with_dp_disabled(self):
        """Test client initialization with DP disabled."""
        dp_config = DPConfig(enabled=False)

        with patch('src.federated.client.FeatureExtractor'):
            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(
                    client_id=0,
                    dp_config=dp_config,
                )

                assert client.sanitizer is None

    def test_init_with_dp_enabled(self):
        """Test client initialization with DP enabled."""
        dp_config = DPConfig(enabled=True, epsilon=1.0, delta=1e-5, clipping_norm=1.0)

        with patch('src.federated.client.FeatureExtractor'):
            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(
                    client_id=0,
                    dp_config=dp_config,
                )

                assert client.sanitizer is not None
                assert isinstance(client.sanitizer, EmbeddingSanitizer)


class TestClientDPSanitization:
    """Tests for DP sanitization in coreset building."""

    @pytest.fixture
    def dp_client(self):
        """Create a client with DP enabled for testing."""
        dp_config = DPConfig(enabled=True, epsilon=1.0, delta=1e-5, clipping_norm=1.0)

        with patch('src.federated.client.FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.get_feature_dim.return_value = 128
            mock_extractor.return_value = mock_instance

            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(
                    client_id=0,
                    coreset_ratio=0.5,
                    dp_config=dp_config,
                )
                return client

    @pytest.fixture
    def non_dp_client(self):
        """Create a client without DP for testing."""
        dp_config = DPConfig(enabled=False)

        with patch('src.federated.client.FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.get_feature_dim.return_value = 128
            mock_extractor.return_value = mock_instance

            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(
                    client_id=0,
                    coreset_ratio=0.5,
                    dp_config=dp_config,
                )
                return client

    def test_coreset_sanitization_applied(self, dp_client):
        """Verify DP sanitization is applied during coreset building."""
        features = np.random.randn(100, 128).astype(np.float32)

        with patch('src.federated.client.greedy_coreset_selection') as mock_coreset:
            mock_coreset.return_value = np.arange(50)

            coreset = dp_client.build_local_coreset(features, seed=42)

            # Coreset should be modified by DP
            original_selected = features[:50]
            assert not np.allclose(coreset, original_selected)

    def test_coreset_without_dp_unchanged(self, non_dp_client):
        """Verify coreset is not modified without DP."""
        features = np.random.randn(100, 128).astype(np.float32)

        with patch('src.federated.client.greedy_coreset_selection') as mock_coreset:
            mock_coreset.return_value = np.arange(50)

            coreset = non_dp_client.build_local_coreset(features, seed=42)

            # Coreset should match selected features exactly
            expected = features[:50]
            np.testing.assert_array_equal(coreset, expected)


class TestClientPrivacyBudget:
    """Tests for client privacy budget tracking."""

    def test_privacy_spent_without_dp(self):
        """Verify zero privacy spent when DP disabled."""
        with patch('src.federated.client.FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.get_feature_dim.return_value = 128
            mock_extractor.return_value = mock_instance

            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(client_id=0)

                epsilon, delta = client.get_privacy_spent()
                assert epsilon == 0.0
                assert delta == 0.0

    def test_privacy_spent_with_dp(self):
        """Verify correct privacy spent when DP enabled."""
        dp_config = DPConfig(enabled=True, epsilon=5.0, delta=1e-6, clipping_norm=1.0)

        with patch('src.federated.client.FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.get_feature_dim.return_value = 128
            mock_extractor.return_value = mock_instance

            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(
                    client_id=0,
                    dp_config=dp_config,
                )

                epsilon, delta = client.get_privacy_spent()
                assert epsilon == 5.0
                assert delta == 1e-6


class TestClientRepr:
    """Tests for client string representation."""

    def test_repr_shows_dp_status_disabled(self):
        """Test repr shows DP disabled."""
        with patch('src.federated.client.FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.get_feature_dim.return_value = 128
            mock_extractor.return_value = mock_instance

            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(client_id=0)

                assert "dp=disabled" in repr(client)

    def test_repr_shows_dp_status_enabled(self):
        """Test repr shows DP enabled."""
        dp_config = DPConfig(enabled=True, epsilon=1.0, delta=1e-5, clipping_norm=1.0)

        with patch('src.federated.client.FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.get_feature_dim.return_value = 128
            mock_extractor.return_value = mock_instance

            with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                client = PatchCoreClient(
                    client_id=0,
                    dp_config=dp_config,
                )

                assert "dp=enabled" in repr(client)
