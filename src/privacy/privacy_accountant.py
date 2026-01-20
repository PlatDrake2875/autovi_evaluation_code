"""Privacy accountant for tracking differential privacy budget."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class PrivacyExpenditure:
    """Record of a single privacy expenditure."""

    epsilon: float
    delta: float
    round_num: int
    description: str = ""


class PrivacyAccountant:
    """Tracks cumulative privacy expenditure across multiple operations.

    Uses basic composition theorem for privacy accounting:
    - Total epsilon = sum of individual epsilons
    - Total delta = sum of individual deltas

    For tighter bounds, consider advanced composition or Rényi DP
    in future iterations.

    Attributes:
        target_epsilon: Optional maximum allowed epsilon.
        expenditures: List of recorded privacy expenditures.
    """

    def __init__(self, target_epsilon: Optional[float] = None):
        """Initialize the privacy accountant.

        Args:
            target_epsilon: Optional maximum privacy budget.
                If set, check_budget() will return False when exceeded.
        """
        self.target_epsilon = target_epsilon
        self.expenditures: List[PrivacyExpenditure] = []

        if target_epsilon is not None:
            logger.info(f"PrivacyAccountant initialized with target epsilon={target_epsilon}")
        else:
            logger.info("PrivacyAccountant initialized without budget limit")

    def record_expenditure(
        self,
        epsilon: float,
        delta: float,
        round_num: int,
        description: str = "",
    ) -> None:
        """Record a privacy expenditure.

        Args:
            epsilon: Privacy parameter spent.
            delta: Failure probability.
            round_num: Training round number.
            description: Optional description of the operation.
        """
        expenditure = PrivacyExpenditure(
            epsilon=epsilon,
            delta=delta,
            round_num=round_num,
            description=description,
        )
        self.expenditures.append(expenditure)

        total_eps, total_delta = self.get_total_privacy()
        logger.debug(
            f"Recorded privacy expenditure: round={round_num}, "
            f"epsilon={epsilon}, delta={delta}. "
            f"Total: epsilon={total_eps:.4f}, delta={total_delta:.2e}"
        )

        if self.target_epsilon and total_eps > self.target_epsilon:
            logger.warning(
                f"Privacy budget exceeded! Total epsilon={total_eps:.4f} > "
                f"target={self.target_epsilon}"
            )

    def get_total_privacy(self) -> Tuple[float, float]:
        """Get the total privacy spent using basic composition.

        Returns:
            Tuple of (total_epsilon, total_delta).
        """
        if not self.expenditures:
            return (0.0, 0.0)

        total_epsilon = sum(exp.epsilon for exp in self.expenditures)
        total_delta = sum(exp.delta for exp in self.expenditures)

        return (total_epsilon, total_delta)

    def check_budget(self) -> bool:
        """Check if the privacy budget is still available.

        Returns:
            True if under budget or no target set, False if exceeded.
        """
        if self.target_epsilon is None:
            return True

        total_epsilon, _ = self.get_total_privacy()
        return total_epsilon <= self.target_epsilon

    def get_remaining_budget(self) -> Optional[float]:
        """Get the remaining epsilon budget.

        Returns:
            Remaining epsilon, or None if no target set.
        """
        if self.target_epsilon is None:
            return None

        total_epsilon, _ = self.get_total_privacy()
        return max(0.0, self.target_epsilon - total_epsilon)

    def get_report(self) -> Dict:
        """Get a comprehensive privacy report.

        Returns:
            Dictionary with privacy accounting details.
        """
        total_epsilon, total_delta = self.get_total_privacy()

        report = {
            "total_epsilon": total_epsilon,
            "total_delta": total_delta,
            "num_expenditures": len(self.expenditures),
            "target_epsilon": self.target_epsilon,
            "remaining_budget": self.get_remaining_budget(),
            "budget_exceeded": not self.check_budget() if self.target_epsilon else False,
            "expenditures_by_round": {},
        }

        # Group expenditures by round
        for exp in self.expenditures:
            round_key = f"round_{exp.round_num}"
            if round_key not in report["expenditures_by_round"]:
                report["expenditures_by_round"][round_key] = []
            report["expenditures_by_round"][round_key].append(
                {
                    "epsilon": exp.epsilon,
                    "delta": exp.delta,
                    "description": exp.description,
                }
            )

        return report

    def reset(self) -> None:
        """Reset all recorded expenditures."""
        self.expenditures.clear()
        logger.info("Privacy accountant reset")

    def __repr__(self) -> str:
        total_eps, total_delta = self.get_total_privacy()
        return (
            f"PrivacyAccountant(total_epsilon={total_eps:.4f}, "
            f"total_delta={total_delta:.2e}, "
            f"expenditures={len(self.expenditures)})"
        )
