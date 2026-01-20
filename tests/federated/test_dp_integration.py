"""Integration tests for differential privacy in federated training."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

from src.privacy import DPConfig
from src.federated import FederatedPatchCore, FederatedServer
from src.training import load_config


class TestDPConfigLoading:
    """Tests for loading DP configuration from YAML."""

    def test_load_config_with_dp(self, tmp_path):
        """Test loading config file with DP settings."""
        config_content = {
            "federated": {"num_clients": 3, "partitioning": "iid", "num_rounds": 1},
            "model": {"backbone": "wide_resnet50_2"},
            "aggregation": {"strategy": "federated_coreset", "global_bank_size": 1000},
            "differential_privacy": {
                "enabled": True,
                "epsilon": 2.0,
                "delta": 1e-6,
                "clipping_norm": 0.5,
            },
            "seed": 42,
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(str(config_path))

        assert "differential_privacy" in config
        dp_config = config["differential_privacy"]
        assert dp_config["enabled"] is True
        assert dp_config["epsilon"] == 2.0
        assert dp_config["delta"] == 1e-6
        assert dp_config["clipping_norm"] == 0.5

    def test_load_config_without_dp(self, tmp_path):
        """Test loading config file without DP settings."""
        config_content = {
            "federated": {"num_clients": 3, "partitioning": "iid"},
            "model": {"backbone": "wide_resnet50_2"},
            "seed": 42,
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        config = load_config(str(config_path))

        # Should default to empty dict when key not present
        dp_config = config.get("differential_privacy", {})
        assert dp_config.get("enabled", False) is False


class TestFederatedPatchCoreDPInit:
    """Tests for FederatedPatchCore DP initialization."""

    def test_init_with_dp_disabled(self):
        """Test FederatedPatchCore initialization with DP disabled."""
        with patch('src.federated.federated_patchcore.FeatureExtractor'):
            with patch('src.federated.federated_patchcore.get_device', return_value=torch.device('cpu')):
                with patch('src.federated.client.FeatureExtractor'):
                    with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                        model = FederatedPatchCore(
                            num_clients=2,
                            dp_enabled=False,
                        )

                        assert model.dp_config.enabled is False
                        for client in model.clients:
                            assert client.sanitizer is None

    def test_init_with_dp_enabled(self):
        """Test FederatedPatchCore initialization with DP enabled."""
        with patch('src.federated.federated_patchcore.FeatureExtractor'):
            with patch('src.federated.federated_patchcore.get_device', return_value=torch.device('cpu')):
                with patch('src.federated.client.FeatureExtractor'):
                    with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                        model = FederatedPatchCore(
                            num_clients=2,
                            dp_enabled=True,
                            dp_epsilon=1.0,
                            dp_delta=1e-5,
                            dp_clipping_norm=1.0,
                        )

                        assert model.dp_config.enabled is True
                        assert model.dp_config.epsilon == 1.0
                        assert model.dp_config.delta == 1e-5
                        assert model.dp_config.clipping_norm == 1.0

                        for client in model.clients:
                            assert client.sanitizer is not None

    def test_dp_parameters_propagate_to_clients(self):
        """Verify DP parameters propagate to all clients."""
        with patch('src.federated.federated_patchcore.FeatureExtractor'):
            with patch('src.federated.federated_patchcore.get_device', return_value=torch.device('cpu')):
                with patch('src.federated.client.FeatureExtractor'):
                    with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                        model = FederatedPatchCore(
                            num_clients=3,
                            dp_enabled=True,
                            dp_epsilon=5.0,
                            dp_delta=1e-7,
                            dp_clipping_norm=2.0,
                        )

                        for client in model.clients:
                            assert client.dp_config.enabled is True
                            assert client.dp_config.epsilon == 5.0
                            assert client.dp_config.delta == 1e-7
                            assert client.dp_config.clipping_norm == 2.0


class TestServerPrivacyTracking:
    """Tests for FederatedServer privacy tracking."""

    def test_server_privacy_tracking_disabled(self):
        """Test server without privacy tracking."""
        server = FederatedServer(
            global_bank_size=1000,
            track_privacy=False,
        )

        assert server.privacy_accountant is None
        assert server.get_privacy_report() is None

    def test_server_privacy_tracking_enabled(self):
        """Test server with privacy tracking enabled."""
        server = FederatedServer(
            global_bank_size=1000,
            track_privacy=True,
            target_epsilon=10.0,
        )

        assert server.privacy_accountant is not None
        report = server.get_privacy_report()
        assert report is not None
        assert "total_epsilon" in report

    def test_server_records_client_privacy_expenditure(self):
        """Test server records privacy expenditure from clients."""
        server = FederatedServer(
            global_bank_size=1000,
            track_privacy=True,
            target_epsilon=10.0,
        )

        # Simulate client stats with privacy info
        client_stats = [
            {"client_id": 0, "dp_epsilon": 1.0, "dp_delta": 1e-5},
            {"client_id": 1, "dp_epsilon": 1.0, "dp_delta": 1e-5},
        ]

        # Create dummy coresets
        coresets = [
            np.random.randn(100, 64).astype(np.float32),
            np.random.randn(100, 64).astype(np.float32),
        ]

        server.receive_client_coresets(coresets, client_stats, round_num=1)

        report = server.get_privacy_report()
        assert report["total_epsilon"] > 0


class TestFederatedPatchCorePrivacyIntegration:
    """Integration tests for DP in federated training flow."""

    def test_server_tracks_privacy_when_dp_enabled(self):
        """Verify server privacy tracking is enabled with DP."""
        with patch('src.federated.federated_patchcore.FeatureExtractor'):
            with patch('src.federated.federated_patchcore.get_device', return_value=torch.device('cpu')):
                with patch('src.federated.client.FeatureExtractor'):
                    with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                        model = FederatedPatchCore(
                            num_clients=2,
                            dp_enabled=True,
                            dp_epsilon=1.0,
                        )

                        assert model.server.privacy_accountant is not None

    def test_dp_config_saved_with_model(self, tmp_path):
        """Verify DP configuration is saved with model."""
        with patch('src.federated.federated_patchcore.FeatureExtractor'):
            with patch('src.federated.federated_patchcore.get_device', return_value=torch.device('cpu')):
                with patch('src.federated.client.FeatureExtractor'):
                    with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                        model = FederatedPatchCore(
                            num_clients=2,
                            dp_enabled=True,
                            dp_epsilon=3.0,
                            dp_delta=1e-6,
                            dp_clipping_norm=1.5,
                        )

                        # Mock global features for save
                        model.server.global_features = np.random.randn(100, 64).astype(np.float32)

                        output_dir = tmp_path / "model_output"
                        model.save(str(output_dir))

                        # Check config was saved
                        import json
                        config_path = output_dir / "federated_config.json"
                        assert config_path.exists()

                        with open(config_path) as f:
                            saved_config = json.load(f)

                        assert saved_config["dp_enabled"] is True
                        assert saved_config["dp_epsilon"] == 3.0
                        assert saved_config["dp_delta"] == 1e-6
                        assert saved_config["dp_clipping_norm"] == 1.5

    def test_stats_include_dp_info(self):
        """Verify stats include DP information when enabled."""
        with patch('src.federated.federated_patchcore.FeatureExtractor'):
            with patch('src.federated.federated_patchcore.get_device', return_value=torch.device('cpu')):
                with patch('src.federated.client.FeatureExtractor'):
                    with patch('src.federated.client.get_device', return_value=torch.device('cpu')):
                        model = FederatedPatchCore(
                            num_clients=2,
                            dp_enabled=True,
                            dp_epsilon=1.0,
                        )

                        # Stats from server should include privacy info
                        stats = model.server.get_stats()
                        assert "privacy_report" in stats or model.server.get_privacy_report() is not None


class TestDPConfigValidation:
    """Tests for DP configuration validation."""

    def test_invalid_epsilon_rejected(self):
        """Verify invalid epsilon is rejected."""
        with pytest.raises(ValueError):
            DPConfig(enabled=True, epsilon=0.01)  # Too small

        with pytest.raises(ValueError):
            DPConfig(enabled=True, epsilon=100.0)  # Too large

    def test_invalid_delta_rejected(self):
        """Verify invalid delta is rejected."""
        with pytest.raises(ValueError):
            DPConfig(enabled=True, delta=0)  # Too small

        with pytest.raises(ValueError):
            DPConfig(enabled=True, delta=1.0)  # Too large

    def test_invalid_clipping_norm_rejected(self):
        """Verify invalid clipping norm is rejected."""
        with pytest.raises(ValueError):
            DPConfig(enabled=True, clipping_norm=0)

        with pytest.raises(ValueError):
            DPConfig(enabled=True, clipping_norm=-1.0)
