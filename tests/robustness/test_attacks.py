"""Tests for ModelPoisoningAttack class."""

import numpy as np
import pytest

from src.robustness.attacks import ModelPoisoningAttack


class TestModelPoisoningAttackInit:
    """Tests for ModelPoisoningAttack initialization."""

    def test_default_parameters(self):
        """Test default parameter values."""
        attack = ModelPoisoningAttack()
        assert attack.attack_type == "scaling"
        assert attack.scale_factor == 100.0
        assert attack.noise_std == 10.0
        assert attack.seed is None

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        attack = ModelPoisoningAttack(
            attack_type="noise",
            scale_factor=50.0,
            noise_std=5.0,
            seed=123,
        )
        assert attack.attack_type == "noise"
        assert attack.scale_factor == 50.0
        assert attack.noise_std == 5.0
        assert attack.seed == 123

    def test_valid_attack_types(self):
        """Test all valid attack types can be initialized."""
        for attack_type in ["scaling", "noise", "sign_flip"]:
            attack = ModelPoisoningAttack(attack_type=attack_type)
            assert attack.attack_type == attack_type

    def test_invalid_attack_type(self):
        """Test that invalid attack type raises ValueError."""
        with pytest.raises(ValueError, match="attack_type must be one of"):
            ModelPoisoningAttack(attack_type="invalid")

    def test_invalid_attack_type_case_sensitive(self):
        """Test that attack type is case sensitive."""
        with pytest.raises(ValueError, match="attack_type must be one of"):
            ModelPoisoningAttack(attack_type="SCALING")


class TestScalingAttack:
    """Tests for scaling attack."""

    def test_scaling_attack_multiplies_values(self):
        """Test that scaling attack multiplies by scale_factor."""
        scale_factor = 10.0
        attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=scale_factor)

        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        client_updates = [original.copy()]

        result = attack.apply(client_updates, malicious_indices=[0])

        expected = original * scale_factor
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_scaling_attack_preserves_shape(self):
        """Test that scaling attack preserves array shape."""
        attack = ModelPoisoningAttack(attack_type="scaling")

        client_updates = [np.random.randn(100, 64)]
        result = attack.apply(client_updates, malicious_indices=[0])

        assert result[0].shape == (100, 64)


class TestNoiseAttack:
    """Tests for noise attack."""

    def test_noise_attack_adds_noise(self):
        """Test that noise attack modifies values."""
        attack = ModelPoisoningAttack(attack_type="noise", noise_std=1.0, seed=42)

        original = np.zeros((100, 64))
        client_updates = [original.copy()]

        result = attack.apply(client_updates, malicious_indices=[0])

        # Result should be different from original
        assert not np.allclose(result[0], original)

    def test_noise_attack_variance(self):
        """Test that noise has expected variance."""
        noise_std = 5.0
        attack = ModelPoisoningAttack(attack_type="noise", noise_std=noise_std, seed=42)

        original = np.zeros((10000, 64))
        client_updates = [original.copy()]

        result = attack.apply(client_updates, malicious_indices=[0])

        # Noise should have std close to noise_std
        measured_std = np.std(result[0])
        assert abs(measured_std - noise_std) / noise_std < 0.1  # 10% tolerance

    def test_noise_attack_reproducibility(self):
        """Test that same seed produces same noise."""
        attack1 = ModelPoisoningAttack(attack_type="noise", seed=42)
        attack2 = ModelPoisoningAttack(attack_type="noise", seed=42)

        original = np.zeros((100, 64))
        client_updates = [original.copy()]

        result1 = attack1.apply([original.copy()], malicious_indices=[0])
        result2 = attack2.apply([original.copy()], malicious_indices=[0])

        np.testing.assert_array_equal(result1[0], result2[0])


class TestSignFlipAttack:
    """Tests for sign flip attack."""

    def test_sign_flip_negates_values(self):
        """Test that sign flip attack negates values."""
        attack = ModelPoisoningAttack(attack_type="sign_flip")

        original = np.array([[1.0, -2.0], [3.0, -4.0]])
        client_updates = [original.copy()]

        result = attack.apply(client_updates, malicious_indices=[0])

        expected = -original
        np.testing.assert_array_equal(result[0], expected)

    def test_sign_flip_double_application(self):
        """Test that applying sign flip twice returns original."""
        attack = ModelPoisoningAttack(attack_type="sign_flip")

        original = np.random.randn(50, 32)
        client_updates = [original.copy()]

        result1 = attack.apply(client_updates, malicious_indices=[0])
        result2 = attack.apply(result1, malicious_indices=[0])

        np.testing.assert_array_almost_equal(result2[0], original)


class TestAttackApplication:
    """Tests for attack application behavior."""

    def test_non_malicious_clients_unchanged(self):
        """Test that non-malicious clients are not modified."""
        attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=100.0)

        client_updates = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0]]),
            np.array([[5.0, 6.0]]),
        ]

        # Only client 1 is malicious
        result = attack.apply(client_updates, malicious_indices=[1])

        # Client 0 and 2 should be unchanged
        np.testing.assert_array_equal(result[0], client_updates[0])
        np.testing.assert_array_equal(result[2], client_updates[2])

        # Client 1 should be scaled
        np.testing.assert_array_equal(result[1], client_updates[1] * 100.0)

    def test_original_arrays_unchanged(self):
        """Test that original arrays are not modified."""
        attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=100.0)

        original = np.array([[1.0, 2.0]])
        original_copy = original.copy()
        client_updates = [original]

        attack.apply(client_updates, malicious_indices=[0])

        # Original should be unchanged
        np.testing.assert_array_equal(original, original_copy)

    def test_empty_client_list(self):
        """Test with empty client list."""
        attack = ModelPoisoningAttack()
        result = attack.apply([], malicious_indices=[])
        assert result == []

    def test_empty_malicious_indices(self):
        """Test with no malicious clients."""
        attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=100.0)

        client_updates = [np.array([[1.0, 2.0]])]
        result = attack.apply(client_updates, malicious_indices=[])

        # Should be unchanged
        np.testing.assert_array_equal(result[0], client_updates[0])

    def test_invalid_malicious_index_too_high(self):
        """Test that out-of-bounds index raises ValueError."""
        attack = ModelPoisoningAttack()
        client_updates = [np.array([[1.0, 2.0]])]

        with pytest.raises(ValueError, match="out of bounds"):
            attack.apply(client_updates, malicious_indices=[5])

    def test_invalid_malicious_index_negative(self):
        """Test that negative index raises ValueError."""
        attack = ModelPoisoningAttack()
        client_updates = [np.array([[1.0, 2.0]])]

        with pytest.raises(ValueError, match="out of bounds"):
            attack.apply(client_updates, malicious_indices=[-1])

    def test_multiple_malicious_clients(self):
        """Test attack on multiple malicious clients."""
        attack = ModelPoisoningAttack(attack_type="sign_flip")

        client_updates = [
            np.array([[1.0]]),
            np.array([[2.0]]),
            np.array([[3.0]]),
            np.array([[4.0]]),
        ]

        result = attack.apply(client_updates, malicious_indices=[0, 2])

        np.testing.assert_array_equal(result[0], [[-1.0]])  # Flipped
        np.testing.assert_array_equal(result[1], [[2.0]])   # Unchanged
        np.testing.assert_array_equal(result[2], [[-3.0]]) # Flipped
        np.testing.assert_array_equal(result[3], [[4.0]])   # Unchanged


class TestAttackStats:
    """Tests for get_attack_stats method."""

    def test_attack_stats_scaling(self):
        """Test attack stats for scaling attack."""
        attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=50.0)

        stats = attack.get_attack_stats(num_clients=10, malicious_indices=[1, 3, 5])

        assert stats["attack_type"] == "scaling"
        assert stats["num_clients"] == 10
        assert stats["num_malicious"] == 3
        assert stats["malicious_fraction"] == 0.3
        assert stats["malicious_indices"] == [1, 3, 5]
        assert stats["scale_factor"] == 50.0
        assert stats["noise_std"] is None

    def test_attack_stats_noise(self):
        """Test attack stats for noise attack."""
        attack = ModelPoisoningAttack(attack_type="noise", noise_std=5.0)

        stats = attack.get_attack_stats(num_clients=5, malicious_indices=[0, 1])

        assert stats["attack_type"] == "noise"
        assert stats["num_malicious"] == 2
        assert stats["malicious_fraction"] == 0.4
        assert stats["scale_factor"] is None
        assert stats["noise_std"] == 5.0

    def test_attack_stats_sign_flip(self):
        """Test attack stats for sign flip attack."""
        attack = ModelPoisoningAttack(attack_type="sign_flip")

        stats = attack.get_attack_stats(num_clients=4, malicious_indices=[])

        assert stats["attack_type"] == "sign_flip"
        assert stats["num_malicious"] == 0
        assert stats["malicious_fraction"] == 0.0
        assert stats["scale_factor"] is None
        assert stats["noise_std"] is None

    def test_attack_stats_zero_clients(self):
        """Test attack stats with zero clients."""
        attack = ModelPoisoningAttack()

        stats = attack.get_attack_stats(num_clients=0, malicious_indices=[])

        assert stats["malicious_fraction"] == 0


class TestRepr:
    """Tests for string representation."""

    def test_repr_scaling(self):
        """Test repr for scaling attack."""
        attack = ModelPoisoningAttack(attack_type="scaling", scale_factor=50.0)
        repr_str = repr(attack)
        assert "ModelPoisoningAttack" in repr_str
        assert "scaling" in repr_str
        assert "50" in repr_str

    def test_repr_noise(self):
        """Test repr for noise attack."""
        attack = ModelPoisoningAttack(attack_type="noise", noise_std=5.0)
        repr_str = repr(attack)
        assert "ModelPoisoningAttack" in repr_str
        assert "noise" in repr_str
        assert "5" in repr_str

    def test_repr_sign_flip(self):
        """Test repr for sign flip attack."""
        attack = ModelPoisoningAttack(attack_type="sign_flip")
        repr_str = repr(attack)
        assert "ModelPoisoningAttack" in repr_str
        assert "sign_flip" in repr_str
