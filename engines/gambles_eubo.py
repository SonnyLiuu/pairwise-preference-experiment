# engines/gambles_eubo.py

"""
Gambles block using MC EUBO for pairwise query selection.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import qExpectedUtilityOfBestOption
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf


class GamblesEUBOTrial:
    """Trial container for the gambles–EUBO block."""

    def __init__(
        self,
        trial_in_block: int,
        block_label: str,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
    ) -> None:
        self.trial_in_block = trial_in_block
        self.block_label = block_label

        self.left_features = left_features
        self.right_features = right_features

        self.left_display = GamblesEUBOEngine.format_gamble(left_features)
        self.right_display = GamblesEUBOEngine.format_gamble(right_features)

        self.model_p_left: Optional[float] = None
        self.model_p_right: Optional[float] = None


class GamblesEUBOEngine:
    """Engine for the gambles–EUBO block."""

    def __init__(self, block_label: str = "gambles_eubo") -> None:
        self.block_label = block_label

        self.dtype = torch.float64
        self.dim = 3
        self.bounds = torch.tensor(
            [[0.05,   1.0,    1.0],
             [0.95, 1000.0, 1000.0]],
            dtype=self.dtype,
        )

        self.num_queries = 20
        self.total_trials = self.num_queries + 1

        self.X: Optional[torch.Tensor] = None
        self.comps: Optional[torch.Tensor] = None
        self.model: Optional[PairwiseGP] = None

        self.trial_in_block = 0
        self.current_pair: Optional[torch.Tensor] = None
        self._done = False

        self.num_restarts = 12
        self.raw_samples = 1024
        self.num_mc_samples = 128

    @staticmethod
    def format_gamble(g: torch.Tensor) -> str:
        """Format [p_win, win_amount, loss_amount] as display string."""
        p = float(g[0].item())
        w = float(g[1].item())
        l = float(g[2].item())
        p_win = int(round(p * 100))
        p_lose = 100 - p_win
        return (
            f"{p_win}% chance of +${w:.0f}\n"
            f"{p_lose}% chance of -${l:.0f}"
        )

    # Engine interface

    def start(self) -> GamblesEUBOTrial:
        if self._done:
            raise RuntimeError("Block already finished.")

        span = self.bounds[1] - self.bounds[0]
        init = self.bounds[0] + span * torch.rand(2, self.dim, dtype=self.dtype)

        if torch.allclose(init[0], init[1]):
            init[1] = self.bounds[0] + span * torch.rand(self.dim, dtype=self.dtype)

        self.current_pair = init
        self.trial_in_block = 1

        return GamblesEUBOTrial(
            trial_in_block=self.trial_in_block,
            block_label=self.block_label,
            left_features=init[0],
            right_features=init[1],
        )

    def next_trial(self) -> Optional[GamblesEUBOTrial]:
        if self._done:
            return None

        if self.model is None or self.X is None or self.comps is None:
            raise RuntimeError("next_trial() called before GP is initialized.")

        if self.trial_in_block >= self.total_trials:
            self._done = True
            return None

        # MC sampler for qEUBO (handles BoTorch API differences)
        try:
            sampler = SobolQMCNormalSampler(num_samples=self.num_mc_samples)
        except TypeError:
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.num_mc_samples])
            )

        acq = qExpectedUtilityOfBestOption(pref_model=self.model, sampler=sampler)

        next_X, _ = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=2,
            sequential=True,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )

        if torch.allclose(next_X[0], next_X[1]):
            span = self.bounds[1] - self.bounds[0]
            next_X[1] = (
                self.bounds[0]
                + span * torch.rand(self.dim, dtype=self.dtype)
            )

        self.current_pair = next_X
        self.trial_in_block += 1

        return GamblesEUBOTrial(
            trial_in_block=self.trial_in_block,
            block_label=self.block_label,
            left_features=next_X[0],
            right_features=next_X[1],
        )

    def update(self, choice: str) -> None:
        if choice not in ("left", "right"):
            raise ValueError(f"Invalid choice: {choice!r}")
        if self.current_pair is None:
            raise RuntimeError("update() called with no current pair.")

        pair = self.current_pair

        if self.X is None or self.comps is None or self.model is None:
            self.X = pair.clone()
            if choice == "left":
                self.comps = torch.tensor([[0, 1]], dtype=torch.long)
            else:
                self.comps = torch.tensor([[1, 0]], dtype=torch.long)
        else:
            n = self.X.shape[0]
            self.X = torch.cat([self.X, pair], dim=0)
            if choice == "left":
                new_comp = torch.tensor([[n, n + 1]], dtype=torch.long)
            else:
                new_comp = torch.tensor([[n + 1, n]], dtype=torch.long)
            self.comps = torch.cat([self.comps, new_comp], dim=0)

        self.model = PairwiseGP(
            self.X,
            self.comps,
            input_transform=Normalize(d=self.dim, bounds=self.bounds),
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        fit_gpytorch_mll(mll)

    def is_done(self) -> bool:
        return self._done
