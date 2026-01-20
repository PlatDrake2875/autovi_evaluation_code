"""Tests for PrivacyAccountant class."""

import pytest

from src.privacy import PrivacyAccountant


class TestPrivacyAccountantInit:
    """Tests for PrivacyAccountant initialization."""

    def test_init_without_target(self):
        """Test initialization without target budget."""
        accountant = PrivacyAccountant()
        assert accountant.target_epsilon is None
        assert len(accountant.expenditures) == 0

    def test_init_with_target(self):
        """Test initialization with target budget."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        assert accountant.target_epsilon == 10.0
        assert len(accountant.expenditures) == 0


class TestRecordExpenditure:
    """Tests for recording privacy expenditures."""

    def test_record_single_expenditure(self):
        """Test recording a single expenditure."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)

        assert len(accountant.expenditures) == 1
        assert accountant.expenditures[0].epsilon == 1.0
        assert accountant.expenditures[0].delta == 1e-5
        assert accountant.expenditures[0].round_num == 1

    def test_record_multiple_expenditures(self):
        """Test recording multiple expenditures."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2)
        accountant.record_expenditure(epsilon=0.5, delta=1e-6, round_num=3)

        assert len(accountant.expenditures) == 3

    def test_record_with_description(self):
        """Test recording expenditure with description."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(
            epsilon=1.0, delta=1e-5, round_num=1, description="Client coreset sanitization"
        )

        assert accountant.expenditures[0].description == "Client coreset sanitization"


class TestTotalPrivacy:
    """Tests for total privacy computation."""

    def test_total_privacy_empty(self):
        """Test total privacy when no expenditures recorded."""
        accountant = PrivacyAccountant()
        total_epsilon, total_delta = accountant.get_total_privacy()

        assert total_epsilon == 0.0
        assert total_delta == 0.0

    def test_total_privacy_single(self):
        """Test total privacy with single expenditure."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)

        total_epsilon, total_delta = accountant.get_total_privacy()
        assert total_epsilon == 1.0
        assert total_delta == 1e-5

    def test_total_privacy_composition(self):
        """Test basic composition theorem (sum of epsilons and deltas)."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2)
        accountant.record_expenditure(epsilon=0.5, delta=1e-6, round_num=3)

        total_epsilon, total_delta = accountant.get_total_privacy()
        assert total_epsilon == 3.5  # 1.0 + 2.0 + 0.5
        assert total_delta == pytest.approx(2.1e-5)  # 1e-5 + 1e-5 + 1e-6


class TestBudgetChecking:
    """Tests for budget checking functionality."""

    def test_check_budget_no_target(self):
        """Test check_budget returns True when no target set."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(epsilon=100.0, delta=1e-5, round_num=1)

        assert accountant.check_budget() is True

    def test_check_budget_under_limit(self):
        """Test check_budget returns True when under target."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2)

        assert accountant.check_budget() is True

    def test_check_budget_at_limit(self):
        """Test check_budget returns True when exactly at target."""
        accountant = PrivacyAccountant(target_epsilon=3.0)
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2)

        assert accountant.check_budget() is True

    def test_check_budget_exceeded(self):
        """Test check_budget returns False when over target."""
        accountant = PrivacyAccountant(target_epsilon=2.0)
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2)

        assert accountant.check_budget() is False


class TestRemainingBudget:
    """Tests for remaining budget calculation."""

    def test_remaining_budget_no_target(self):
        """Test remaining budget is None when no target set."""
        accountant = PrivacyAccountant()
        assert accountant.get_remaining_budget() is None

    def test_remaining_budget_full(self):
        """Test remaining budget equals target when no spending."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        assert accountant.get_remaining_budget() == 10.0

    def test_remaining_budget_partial(self):
        """Test remaining budget after partial spending."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        accountant.record_expenditure(epsilon=3.0, delta=1e-5, round_num=1)

        assert accountant.get_remaining_budget() == 7.0

    def test_remaining_budget_exceeded(self):
        """Test remaining budget is zero when exceeded."""
        accountant = PrivacyAccountant(target_epsilon=5.0)
        accountant.record_expenditure(epsilon=10.0, delta=1e-5, round_num=1)

        assert accountant.get_remaining_budget() == 0.0


class TestReport:
    """Tests for privacy report generation."""

    def test_report_empty(self):
        """Test report with no expenditures."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        report = accountant.get_report()

        assert report["total_epsilon"] == 0.0
        assert report["total_delta"] == 0.0
        assert report["num_expenditures"] == 0
        assert report["target_epsilon"] == 10.0
        assert report["remaining_budget"] == 10.0
        assert report["budget_exceeded"] is False
        assert report["expenditures_by_round"] == {}

    def test_report_with_expenditures(self):
        """Test report with recorded expenditures."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1, description="Round 1")
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2, description="Round 2")

        report = accountant.get_report()

        assert report["total_epsilon"] == 3.0
        assert report["num_expenditures"] == 2
        assert report["remaining_budget"] == 7.0
        assert "round_1" in report["expenditures_by_round"]
        assert "round_2" in report["expenditures_by_round"]

    def test_report_groups_by_round(self):
        """Test report groups expenditures by round."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=0.5, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2)

        report = accountant.get_report()

        assert len(report["expenditures_by_round"]["round_1"]) == 2
        assert len(report["expenditures_by_round"]["round_2"]) == 1


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_expenditures(self):
        """Test reset clears all recorded expenditures."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)
        accountant.record_expenditure(epsilon=2.0, delta=1e-5, round_num=2)

        accountant.reset()

        assert len(accountant.expenditures) == 0
        assert accountant.get_total_privacy() == (0.0, 0.0)

    def test_reset_preserves_target(self):
        """Test reset preserves target epsilon."""
        accountant = PrivacyAccountant(target_epsilon=10.0)
        accountant.record_expenditure(epsilon=5.0, delta=1e-5, round_num=1)

        accountant.reset()

        assert accountant.target_epsilon == 10.0


class TestRepr:
    """Tests for string representation."""

    def test_repr_format(self):
        """Test repr contains expected information."""
        accountant = PrivacyAccountant()
        accountant.record_expenditure(epsilon=1.0, delta=1e-5, round_num=1)

        repr_str = repr(accountant)
        assert "total_epsilon=" in repr_str
        assert "total_delta=" in repr_str
        assert "expenditures=1" in repr_str
