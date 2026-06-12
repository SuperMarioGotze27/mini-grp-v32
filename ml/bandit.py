"""
factor_bandit.py — Multi-Armed Bandits for Dynamic Factor Weight Allocation.

This module implements three bandit algorithms for dynamically adjusting
factor weights across four style dimensions (Value, Quality, Growth, Momentum)
in a cross-market quantitative stock-selection system (Mini-GRP).

Algorithms
----------
1. EpsilonGreedyFactorBandit — ε-greedy explore/exploit
2. UCBFactorBandit — Upper Confidence Bound
3. ContextualLinUCBandit — Contextual Linear UCB with market regime features

Course References
-----------------
All algorithms map directly to topics covered in the Decision Analytics
lecture series on Multi-Armed Bandits:
    • Lecture 4 — The Explore-Exploitation Dilemma (ε-greedy)
    • Lecture 5 — UCB and O(sqrt(N)) regret bounds
    • Lecture 5 Extension III — Contextual LinUCB

Author : Mini-GRP Quant Team
Date   : 2025-06-18
"""

from __future__ import annotations

__all__ = [
    "EpsilonGreedyFactorBandit",
    "UCBFactorBandit",
    "ContextualLinUCBandit",
]

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax with temperature scaling."""
    x_shifted = (x - np.max(x)) / max(temperature, 1e-8)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def _ensure_dirichlet_like(weights: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip negatives, add eps, and normalise so the vector sums to 1."""
    w = np.maximum(weights, 0.0) + eps
    return w / w.sum()


# ---------------------------------------------------------------------------
# 1. ε-Greedy Bandit
# ---------------------------------------------------------------------------

class EpsilonGreedyFactorBandit:
    """ε-Greedy Multi-Armed Bandit for dynamic factor weight allocation.

    Treats the four style dimensions (Value / Quality / Growth / Momentum) as
    four independent arms.  Each month the bandit observes the forward-return
    Information Coefficient (IC) for every arm, updates its empirical mean
    reward, and stochastically chooses between
        * exploration  — with probability ``epsilon`` a random weight vector
                         is drawn from a flat Dirichlet distribution;
        * exploitation — with probability ``1 - epsilon`` the arm with the
                         highest historical average IC receives the lion's
                         share of weight.

    Parameters
    ----------
    n_arms : int
        Number of arms (default 4 for V/Q/G/M).
    epsilon : float
        Exploration probability in [0, 1] (default 0.15).
    arm_names : list of str, optional
        Human-readable labels for each arm.

    Attributes
    ----------
    counts : np.ndarray, shape (n_arms,)
        Number of times each arm has been pulled / updated.
    values : np.ndarray, shape (n_arms,)
        Incremental average reward (IC) for each arm.
    total_reward : float
        Cumulative reward received so far.
    optimal_reward_history : list of float
        History of the *best* reward observed at each round (for regret calc).

    Course Concept
    --------------
    Explore-Exploitation Dilemma — Lecture 4.
    The ε-greedy rule is the simplest frequentist strategy for balancing
    exploration against exploitation.  Its regret is linear in T because the
    random exploration never vanishes, but it is intuitive and state-less.
    """

    def __init__(
        self,
        n_arms: int = 4,
        epsilon: float = 0.15,
        arm_names: Optional[List[str]] = None,
    ) -> None:
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must lie in [0, 1]")
        self.n_arms: int = n_arms
        self.epsilon: float = epsilon
        self.arm_names: List[str] = arm_names or [f"Arm_{i}" for i in range(n_arms)]

        # sufficient statistics
        self.counts: np.ndarray = np.zeros(n_arms, dtype=np.float64)
        self.values: np.ndarray = np.zeros(n_arms, dtype=np.float64)
        self.total_reward: float = 0.0
        self.t: int = 0  # total number of update rounds

        # regret bookkeeping
        self.reward_history: List[float] = []
        self.optimal_reward_history: List[float] = []

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def select_weights(self) -> np.ndarray:
        """Return a 4-dim weight vector under the ε-greedy policy.

        With probability ``epsilon`` we *explore*: sample weights uniformly
        at random and normalise.  With probability ``1 - epsilon`` we
        *exploit*: give maximum weight to the arm with the highest empirical
        mean IC, while still assigning a small residual to the others so that
        the resulting vector stays well-conditioned for downstream optimisers.

        Returns
        -------
        weights : np.ndarray, shape (n_arms,)
            Non-negative weights that sum to 1.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: random weights from Dirichlet(1,...,1)
            raw = np.random.rand(self.n_arms)
            return raw / raw.sum()

        # Exploitation: boost the best arm, keep small mass on the rest
        best_arm = int(np.argmax(self.values))
        weights = np.full(self.n_arms, 0.05 / (self.n_arms - 1))
        weights[best_arm] = 0.95
        return weights

    def update(self, arm_idx: int, reward: float) -> None:
        """Observe a reward for *one* arm and update its sufficient statistics.

        Uses the incremental update rule (Welford-style) so that no history
        buffer is required:

            Q_new = Q_old + (r - Q_old) / n

        Parameters
        ----------
        arm_idx : int
            Index of the arm that produced ``reward``.
        reward : float
            The realised reward — typically the rank Information Coefficient
            (IC) or the quantile spread for that factor dimension.
        """
        if not (0 <= arm_idx < self.n_arms):
            raise IndexError(f"arm_idx {arm_idx} out of range [0, {self.n_arms})")

        self.t += 1
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]

        # incremental mean update — Q_new = Q_old + (r - Q_old) / n
        old_val = self.values[arm_idx]
        self.values[arm_idx] = old_val + (reward - old_val) / n

        self.total_reward += reward
        self.reward_history.append(reward)

        # for regret we need the best reward that *could* have been obtained
        # at this round: assume the oracle knew the best arm's realised reward
        best_possible = max(self.values)
        self.optimal_reward_history.append(best_possible)

    def get_stats(self) -> Dict:
        """Return a dictionary of current diagnostic statistics.

        Returns
        -------
        stats : dict
            * ``counts``           — pull counts per arm
            * ``values``           — empirical mean reward per arm
            * ``total_reward``     — cumulative reward
            * ``cumulative_regret``— sum of (optimal - actual) rewards
            * ``arm_names``        — human-readable arm labels
        """
        cumulative_regret = 0.0
        if len(self.reward_history) > 0 and len(self.optimal_reward_history) > 0:
            # approximate: optimal_reward_history stores the best *mean* at each step
            actual_means = np.array(self.reward_history)
            optimal_means = np.array(self.optimal_reward_history)
            cumulative_regret = float(np.sum(optimal_means - actual_means))

        return {
            "algorithm": "EpsilonGreedy",
            "epsilon": self.epsilon,
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "total_reward": self.total_reward,
            "cumulative_regret": cumulative_regret,
            "arm_names": self.arm_names.copy(),
        }


# ---------------------------------------------------------------------------
# 2. UCB Bandit
# ---------------------------------------------------------------------------

class UCBFactorBandit:
    """Upper Confidence Bound Bandit for factor weight allocation.

    UCB naturally balances exploration and exploitation by adding an
    optimism bonus to each arm's empirical mean.  Arms that have been
    pulled fewer times receive a larger bonus, encouraging exploration of
    uncertain actions.  As the number of pulls grows the bonus shrinks
    (proportional to ``1/sqrt(count)``) and the policy converges to
    greedy exploitation.

    The per-arm UCB score is

        UCB_i = mean_reward_i + alpha * sqrt(2 * ln(total_pulls) / count_i)

    where ``alpha`` is a tunable exploration constant.

    Parameters
    ----------
    n_arms : int
        Number of arms (default 4).
    alpha : float
        Exploration multiplier (default 1.0).  Larger ``alpha`` → more
        exploration.
    arm_names : list of str, optional
        Human-readable labels.

    Attributes
    ----------
    counts : np.ndarray, shape (n_arms,)
        Pull counts.
    values : np.ndarray, shape (n_arms,)
        Empirical mean rewards.
    total_pulls : int
        Sum of all counts.

    Course Concept
    --------------
    UCB achieves O(sqrt(K * T * ln T)) expected regret — Lecture 5.
    The optimism-in-the-face-of-uncertainty principle guarantees that the
    bonus captures the maximal plausible deviation of the sample mean from
    the true mean (Hoeffding bound), so sub-optimal arms are only chosen
    logarithmically often.
    """

    def __init__(
        self,
        n_arms: int = 4,
        alpha: float = 1.0,
        arm_names: Optional[List[str]] = None,
    ) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.n_arms: int = n_arms
        self.alpha: float = alpha
        self.arm_names: List[str] = arm_names or [f"Arm_{i}" for i in range(n_arms)]

        self.counts: np.ndarray = np.zeros(n_arms, dtype=np.float64)
        self.values: np.ndarray = np.zeros(n_arms, dtype=np.float64)
        self.total_pulls: int = 0

        # regret bookkeeping
        self.reward_history: List[float] = []
        self.best_mean_history: List[float] = []

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def select_weights(self) -> np.ndarray:
        """Compute UCB scores and return a normalised weight vector.

        For arms that have *never* been pulled, the exploration bonus is
        infinite, so they are guaranteed to be tried first (automatic
        initialisation).  After every arm has at least one observation the
        scores are finite and the weights are derived via a softmax over
        the UCB scores.

        Returns
        -------
        weights : np.ndarray, shape (n_arms,)
            Non-negative weights summing to 1.
        """
        ucb_scores = np.zeros(self.n_arms, dtype=np.float64)

        for i in range(self.n_arms):
            if self.counts[i] == 0:
                # Force exploration of un-tried arms
                ucb_scores[i] = np.inf
            else:
                mean_reward = self.values[i]
                bonus = self.alpha * np.sqrt(
                    2.0 * np.log(self.total_pulls + 1) / self.counts[i]
                )
                ucb_scores[i] = mean_reward + bonus

        # If any arm is +inf, give it all mass (uniformly among infinities)
        if np.any(np.isinf(ucb_scores)):
            weights = np.zeros(self.n_arms)
            weights[np.isinf(ucb_scores)] = 1.0
            return weights / weights.sum()

        # Softmax over UCB scores → smooth, differentiable weights
        return _softmax(ucb_scores, temperature=1.0)

    def update(self, arm_idx: int, reward: float) -> None:
        """Update the sufficient statistics for a single arm.

        Parameters
        ----------
        arm_idx : int
            Arm that generated ``reward``.
        reward : float
            Realised reward (IC or quantile spread).
        """
        if not (0 <= arm_idx < self.n_arms):
            raise IndexError(f"arm_idx {arm_idx} out of range [0, {self.n_arms})")

        self.total_pulls += 1
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]

        old_val = self.values[arm_idx]
        self.values[arm_idx] = old_val + (reward - old_val) / n

        self.reward_history.append(reward)
        self.best_mean_history.append(float(np.max(self.values)))

    def get_regret_bound(self, horizon: Optional[int] = None) -> float:
        """Return the theoretical UCB regret bound.

        For K arms and horizon N the worst-case regret of UCB satisfies

            R_N ≤ O(sqrt(K * N * ln N))

        Parameters
        ----------
        horizon : int, optional
            Time horizon N.  Defaults to ``self.total_pulls`` if available,
            otherwise falls back to 100.

        Returns
        -------
        bound : float
            Theoretical upper bound on cumulative regret.

        References
        ----------
        Auer, Cesa-Bianchi & Fischer (2002) — Finite-time Analysis of the
        Multi-armed Bandit Problem.  *Machine Learning*, 47, 235-256.
        """
        N = horizon or max(self.total_pulls, 100)
        K = self.n_arms
        return float(np.sqrt(K * N * np.log(N)))

    def get_stats(self) -> Dict:
        """Return diagnostic statistics for the UCB bandit.

        Returns
        -------
        stats : dict
        """
        cumulative_regret = 0.0
        if self.reward_history:
            actual = np.array(self.reward_history)
            optimal = np.array(self.best_mean_history)
            cumulative_regret = float(np.sum(optimal - actual))

        return {
            "algorithm": "UCB",
            "alpha": self.alpha,
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "total_pulls": self.total_pulls,
            "theoretical_regret_bound": self.get_regret_bound(),
            "cumulative_regret": cumulative_regret,
            "arm_names": self.arm_names.copy(),
        }


# ---------------------------------------------------------------------------
# 3. Contextual LinUCB Bandit
# ---------------------------------------------------------------------------

class ContextualLinUCBandit:
    """Contextual Linear UCB Bandit — market regime drives factor weights.

    Rather than maintaining a single scalar mean per arm, each arm maintains
    a *linear* reward model conditioned on a context vector that encodes the
    prevailing market regime:

        context = [1, volatility, market_return, liquidity]^T

    This allows the bandit to learn that, for example, *Momentum* works well
    when volatility is high and trend is strong, while *Value* shines during
    bear-market recoveries.

    Algorithm (per arm *i*)
    -----------------------
    1. Design matrix   :  A_i = Σ_t x_t x_t^T + alpha * I_d
    2. Reward vector   :  b_i = Σ_t x_t * r_{t,i}
    3. Parameter est.  :  theta_i = A_i^{-1} b_i
    4. UCB score       :  s_i = x^T theta_i + alpha * sqrt(x^T A_i^{-1} x)

    The term ``sqrt(x^T A_i^{-1} x)`` is the *predictive standard deviation*
    of the linear model; multiplying by ``alpha`` yields the optimism bonus.

    Parameters
    ----------
    n_arms : int
        Number of arms (default 4 for V/Q/G/M).
    context_dim : int
        Dimension of the context feature vector (default 4: bias + 3
        market features).
    alpha : float
        Exploration–exploitation trade-off parameter (default 1.0).
    arm_names : list of str, optional
        Human-readable labels.

    Attributes
    ----------
    A : dict[int, np.ndarray]
        Design matrix for each arm (d × d).
    b : dict[int, np.ndarray]
        Reward accumulator for each arm (d,).
    theta : dict[int, np.ndarray]
        Latest parameter estimate for each arm (d,).

    Course Concept
    --------------
    Extension III: Contextual LinUCB — Lecture 5.
    This is the linear realisable case of contextual bandits where the
    expected reward of each arm is a linear function of the context.
    The regret bound is O(d * sqrt(T) * ln T) where d is the context
    dimensionality — a significant improvement over context-free UCB when
    the context is informative.

    References
    ----------
    Li et al. (2010) — A Contextual-Bandit Approach to Personalized News
    Article Recommendation.  *WWW*, 661-670.
    """

    def __init__(
        self,
        n_arms: int = 4,
        context_dim: int = 4,
        alpha: float = 1.0,
        arm_names: Optional[List[str]] = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.n_arms: int = n_arms
        self.context_dim: int = context_dim
        self.alpha: float = alpha
        self.arm_names: List[str] = arm_names or [f"Arm_{i}" for i in range(n_arms)]

        # Per-arm linear-model sufficient statistics
        self.A: Dict[int, np.ndarray] = {}
        self.b: Dict[int, np.ndarray] = {}
        self.theta: Dict[int, np.ndarray] = {}

        for i in range(n_arms):
            self.A[i] = np.eye(context_dim, dtype=np.float64) * alpha
            self.b[i] = np.zeros(context_dim, dtype=np.float64)
            self.theta[i] = np.zeros(context_dim, dtype=np.float64)

        self.t: int = 0
        self.reward_history: List[Tuple[int, np.ndarray, float]] = []

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def select_weights(self, context: np.ndarray) -> np.ndarray:
        """Compute context-dependent UCB weights.

        Parameters
        ----------
        context : np.ndarray, shape (context_dim,) or (context_dim, 1)
            Market-regime feature vector, conventionally
            ``[1, volatility, market_return, liquidity]``.  The leading 1
            serves as the bias term.

        Returns
        -------
        weights : np.ndarray, shape (n_arms,)
            Non-negative weights that sum to 1.
        """
        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context dim {x.shape[0]} != expected {self.context_dim}"
            )

        scores = np.zeros(self.n_arms, dtype=np.float64)

        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            self.theta[i] = A_inv @ self.b[i]

            mean_pred = x @ self.theta[i]          # x^T theta_i
            std_pred = np.sqrt(x @ A_inv @ x)      # sqrt(x^T A_i^{-1} x)
            scores[i] = mean_pred + self.alpha * std_pred

        # Softmax over scores → smooth weights
        return _softmax(scores, temperature=1.0)

    def update(self, arm_idx: int, context: np.ndarray, reward: float) -> None:
        """Update the linear model for one arm after observing a contextual reward.

        Parameters
        ----------
        arm_idx : int
            Arm that was played.
        context : np.ndarray, shape (context_dim,)
            The context vector that was active when the arm was played.
        reward : float
            Realised reward for ``arm_idx`` under ``context``.
        """
        if not (0 <= arm_idx < self.n_arms):
            raise IndexError(f"arm_idx {arm_idx} out of range [0, {self.n_arms})")

        x = np.asarray(context, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context dim {x.shape[0]} != expected {self.context_dim}"
            )

        self.t += 1
        # Sherman-Morrison friendly update
        self.A[arm_idx] += np.outer(x, x)          # A += x x^T
        self.b[arm_idx] += x * reward              # b += x * r

        self.reward_history.append((arm_idx, x.copy(), reward))

    def get_regime_weights(self, regimes: List[np.ndarray]) -> pd.DataFrame:
        """Evaluate the recommended weights for a list of market regimes.

        This is useful for *offline inspection*: after training, one can
        query the bandit with prototypical regime vectors (e.g. Bull, Bear,
        Volatile, Range-Bound) and tabulate the resulting factor weights.

        Parameters
        ----------
        regimes : list of np.ndarray
            Each element is a context vector of length ``context_dim``.

        Returns
        -------
        df : pd.DataFrame
            Columns: ``regime_id`` | ``Value`` | ``Quality`` | ``Growth`` |
            ``Momentum``.  One row per supplied regime.
        """
        records: List[Dict] = []
        for rid, ctx in enumerate(regimes):
            w = self.select_weights(ctx)
            row = {"regime_id": rid}
            for name, weight in zip(self.arm_names, w):
                row[name] = round(float(weight), 4)
            records.append(row)
        return pd.DataFrame(records)

    def get_stats(self) -> Dict:
        """Return diagnostic statistics for the contextual bandit.

        Returns
        -------
        stats : dict
            Includes per-arm theta norms, minimum eigenvalue of each A
            matrix (a proxy for parameter identifiability), and the number
            of observations processed.
        """
        theta_norms = {name: float(np.linalg.norm(self.theta[i]))
                       for i, name in enumerate(self.arm_names)}
        min_eigvals = {name: float(np.min(np.linalg.eigvalsh(self.A[i])))
                       for i, name in enumerate(self.arm_names)}

        return {
            "algorithm": "ContextualLinUCB",
            "alpha": self.alpha,
            "context_dim": self.context_dim,
            "n_observations": self.t,
            "theta_norms": theta_norms,
            "min_eigenvalues": min_eigvals,
            "arm_names": self.arm_names.copy(),
        }


# ============================================================================
# Usage Example — Synthetic Data Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("Mini-GRP Factor Bandit — Synthetic Data Demo")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Global settings
    # ------------------------------------------------------------------
    np.random.seed(42)
    N_ROUNDS = 120          # 120 months = 10 years
    ARM_NAMES = ["Value", "Quality", "Growth", "Momentum"]

    # True underlying IC means for the four factors (unknown to agents)
    TRUE_IC_MEANS = np.array([0.03, 0.05, 0.04, 0.06])
    TRUE_IC_STD = 0.08

    print(f"\nSimulation parameters:")
    print(f"  Rounds    : {N_ROUNDS} (months)")
    print(f"  Arms      : {ARM_NAMES}")
    print(f"  True IC μ : {TRUE_IC_MEANS}")
    print(f"  True IC σ : {TRUE_IC_STD}")

    # ==================================================================
    # 1. Epsilon-Greedy Demo
    # ==================================================================
    print("\n" + "-" * 72)
    print("[1] Epsilon-Greedy Bandit Demo")
    print("-" * 72)

    eg_bandit = EpsilonGreedyFactorBandit(
        n_arms=4, epsilon=0.15, arm_names=ARM_NAMES
    )
    eg_weights_history: List[np.ndarray] = []

    for month in range(N_ROUNDS):
        # 1. choose weights
        w = eg_bandit.select_weights()
        eg_weights_history.append(w)

        # 2. observe noisy IC for each arm (simulated market)
        realized_ic = np.random.normal(TRUE_IC_MEANS, TRUE_IC_STD)

        # 3. update each arm with its realised reward
        for arm in range(4):
            eg_bandit.update(arm, realized_ic[arm])

    stats_eg = eg_bandit.get_stats()
    print(f"\nFinal statistics:")
    print(f"  Counts    : {stats_eg['counts']}")
    print(f"  Q-values  : {np.round(stats_eg['values'], 4)}")
    print(f"  Cum. Regret (approx): {stats_eg['cumulative_regret']:.4f}")

    print(f"\nLast 3 monthly weight allocations:")
    for i, w in enumerate(eg_weights_history[-3:], start=N_ROUNDS - 2):
        print(f"  Month {i:3d}: " + " | ".join(
            f"{n}={v:.3f}" for n, v in zip(ARM_NAMES, w)
        ))

    # ==================================================================
    # 2. UCB Bandit Demo
    # ==================================================================
    print("\n" + "-" * 72)
    print("[2] UCB Bandit Demo")
    print("-" * 72)

    ucb_bandit = UCBFactorBandit(n_arms=4, alpha=1.0, arm_names=ARM_NAMES)
    ucb_weights_history: List[np.ndarray] = []

    for month in range(N_ROUNDS):
        w = ucb_bandit.select_weights()
        ucb_weights_history.append(w)

        realized_ic = np.random.normal(TRUE_IC_MEANS, TRUE_IC_STD)
        for arm in range(4):
            ucb_bandit.update(arm, realized_ic[arm])

    stats_ucb = ucb_bandit.get_stats()
    print(f"\nFinal statistics:")
    print(f"  Counts    : {stats_ucb['counts']}")
    print(f"  Q-values  : {np.round(stats_ucb['values'], 4)}")
    print(f"  Theoretical Regret Bound: {stats_ucb['theoretical_regret_bound']:.4f}")
    print(f"  Cum. Regret (approx)    : {stats_ucb['cumulative_regret']:.4f}")

    print(f"\nLast 3 monthly weight allocations:")
    for i, w in enumerate(ucb_weights_history[-3:], start=N_ROUNDS - 2):
        print(f"  Month {i:3d}: " + " | ".join(
            f"{n}={v:.3f}" for n, v in zip(ARM_NAMES, w)
        ))

    # ==================================================================
    # 3. Contextual LinUCB Demo
    # ==================================================================
    print("\n" + "-" * 72)
    print("[3] Contextual LinUCB Bandit Demo")
    print("-" * 72)

    ctx_bandit = ContextualLinUCBandit(
        n_arms=4, context_dim=4, alpha=1.0, arm_names=ARM_NAMES
    )
    ctx_weights_history: List[np.ndarray] = []

    # Define regime-dependent true IC surfaces (each factor has different
    # sensitivities to volatility and market return).
    # Factor i:  IC_i = base_i + beta_vol_i * vol + beta_ret_i * ret + noise
    BETA_VOL = np.array([0.10, -0.05, 0.15, 0.30])   # Momentum loves vol
    BETA_RET = np.array([0.05, 0.08, 0.20, 0.25])   # Growth/Momentum love bull

    # Simulate 120 months with varying regimes
    contexts_history: List[np.ndarray] = []
    for month in range(N_ROUNDS):
        # Random regime features (volatility, market_return, liquidity)
        vol = np.random.uniform(0.05, 0.40)       # annualised vol
        mret = np.random.uniform(-0.03, 0.03)     # monthly return
        liq = np.random.uniform(0.5, 1.5)         # liquidity index
        ctx = np.array([1.0, vol, mret, liq])     # bias term = 1
        contexts_history.append(ctx)

        # 1. choose weights given context
        w = ctx_bandit.select_weights(ctx)
        ctx_weights_history.append(w)

        # 2. regime-dependent realised IC
        regime_ic = (
            TRUE_IC_MEANS
            + BETA_VOL * vol
            + BETA_RET * mret
            + np.random.normal(0, TRUE_IC_STD, size=4)
        )

        # 3. update every arm with the same context but its own reward
        for arm in range(4):
            ctx_bandit.update(arm, ctx, regime_ic[arm])

    stats_ctx = ctx_bandit.get_stats()
    print(f"\nFinal statistics:")
    print(f"  Observations : {stats_ctx['n_observations']}")
    print(f"  Theta norms  : {stats_ctx['theta_norms']}")
    print(f"  Min eigvals  : {stats_ctx['min_eigenvalues']}")

    # ------------------------------------------------------------------
    # Regime inspection table
    # ------------------------------------------------------------------
    print(f"\nRegime-dependent recommended weights:")
    regime_labels = ["Bull_LowVol", "Bull_HighVol", "Bear_LowVol", "Bear_HighVol"]
    regime_vectors = [
        np.array([1.0, 0.10, 0.02, 1.0]),   # Bull, low vol
        np.array([1.0, 0.35, 0.02, 1.0]),   # Bull, high vol
        np.array([1.0, 0.10, -0.02, 0.6]),  # Bear, low vol
        np.array([1.0, 0.35, -0.02, 0.6]),  # Bear, high vol
    ]

    regime_df = ctx_bandit.get_regime_weights(regime_vectors)
    regime_df["regime_label"] = regime_labels
    # reorder columns
    cols = ["regime_label"] + ARM_NAMES
    print("\n" + regime_df[cols].to_string(index=False))

    # ------------------------------------------------------------------
    # Show a few monthly trajectories
    # ------------------------------------------------------------------
    print(f"\nLast 3 monthly weight allocations (with context):")
    for i in range(N_ROUNDS - 3, N_ROUNDS):
        ctx = contexts_history[i]
        w = ctx_weights_history[i]
        print(f"  Month {i+1:3d} | vol={ctx[1]:.2f} ret={ctx[2]:+.3f} liq={ctx[3]:.2f} | "
              + " | ".join(f"{n}={v:.3f}" for n, v in zip(ARM_NAMES, w)))

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 72)
    print("Summary — All 3 bandit classes completed successfully.")
    print("=" * 72)
    print("\nKey take-aways:")
    print("  • ε-Greedy : simple, state-less, linear regret (ε never decays).")
    print("  • UCB      : optimism bonus → O(sqrt(T ln T)) regret.")
    print("  • LinUCB   : context-aware, learns regime-dependent factor premiums.")
