# engines/cars_bald.py

"""
Cars block using BALD for pairwise query selection.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import PairwiseBayesianActiveLearningByDisagreement
from botorch.optim import optimize_acqf


class CarsBALDTrial:
    """Trial container for the cars–BALD block."""

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

        self.left_display = CarsBALDEngine.format_car(left_features)
        self.right_display = CarsBALDEngine.format_car(right_features)

        self.model_p_left: Optional[float] = None
        self.model_p_right: Optional[float] = None


class CarsBALDEngine:
    """Engine for the cars–BALD block."""

    BRANDS = ["Benz", "Toyota", "Ford"]
    COLORS = ["Black", "White", "Red"]
    FUELS = ["Hybrid", "Gas", "Electric"]

    def __init__(self, block_label: str = "cars_bald") -> None:
        self.block_label = block_label

        self.dtype = torch.float64
        self.dim = 3
        self.bounds = torch.tensor(
            [[0.0, 0.0, 0.0],
             [2.0, 2.0, 2.0]],
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

        self.num_restarts = 5
        self.raw_samples = 256
        self.bald_samples = 256

    @staticmethod
    def snap_discrete(x: torch.Tensor) -> torch.Tensor:
        """Round to nearest {0,1,2} per dimension."""
        return x.round().clamp(0.0, 2.0)

    @classmethod
    def format_car(cls, x: torch.Tensor) -> str:
        """Format [brand, color, fuel] as string."""
        ix = cls.snap_discrete(x).to(torch.int64)
        b = cls.BRANDS[ix[0].item()]
        c = cls.COLORS[ix[1].item()]
        f = cls.FUELS[ix[2].item()]
        return f"{f} {c} {b}"

    # Engine interface

    def start(self) -> CarsBALDTrial:
        if self._done:
            raise RuntimeError("Block already finished.")

        init_idx = torch.randint(0, 3, (2, self.dim))
        init = init_idx.to(dtype=self.dtype)

        if torch.all(init[0] == init[1]):
            dim_to_change = torch.randint(0, self.dim, (1,)).item()
            init[1, dim_to_change] = (init[1, dim_to_change] + 1) % 3

        self.current_pair = init
        self.trial_in_block = 1

        return CarsBALDTrial(
            trial_in_block=self.trial_in_block,
            block_label=self.block_label,
            left_features=init[0],
            right_features=init[1],
        )

    def next_trial(self) -> Optional[CarsBALDTrial]:
        if self._done:
            return None

        if self.model is None or self.X is None or self.comps is None:
            raise RuntimeError("next_trial() called before GP is initialized.")

        if self.trial_in_block >= self.total_trials:
            self._done = True
            return None

        acq = PairwiseBayesianActiveLearningByDisagreement(
            pref_model=self.model,
            num_samples=self.bald_samples,
        )

        next_X_cont, _ = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=2,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )

        next_X = self.snap_discrete(next_X_cont).to(self.dtype)

        if torch.all(next_X[0] == next_X[1]):
            levels = torch.arange(3, dtype=self.dtype)
            all_cars = torch.stack(
                torch.meshgrid(levels, levels, levels, indexing="ij"),
                dim=-1,
            ).view(-1, self.dim)
            mask = ~(all_cars == next_X[0]).all(dim=1)
            candidates = all_cars[mask]
            j = torch.randint(0, candidates.size(0), (1,))
            next_X[1] = candidates[j]

        self.current_pair = next_X
        self.trial_in_block += 1

        return CarsBALDTrial(
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
        mll = PairwiseLaplaceMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def is_done(self) -> bool:
        return self._done
