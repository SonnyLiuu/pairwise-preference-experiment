!pip install torch
!pip install botorch
!pip install gpytorch

import math
import torch

# Probability Algorithm
@torch.no_grad()
def predict_pref_proba(model, x_i: torch.Tensor, x_j: torch.Tensor):
    """
    Closed-form probit integration (i ~ GP, ε ~ N(0,1) i.i.d.):
    P(i > j) = Φ( (μ_i - μ_j) / sqrt(var_i + var_j - 2*cov_ij + 2) )
    Returns (p_i, p_j) as Python floats.
    """
    X = torch.stack((x_i, x_j), dim=0) # (2, d), same device/dtype
    post = model.posterior(X)
    m = post.mean.reshape(-1)      # (2,)
    K = post.mvn.covariance_matrix # (2, 2)

    # Δ mean/variance
    mu_delta  = m[0] - m[1]
    var_delta = K[0, 0] + K[1, 1] - 2 * K[0, 1]

    # add probit noise: (ε1 - ε2) ~ N(0, 2)
    eps = torch.finfo(m.dtype).eps
    denom = torch.sqrt(torch.clamp(var_delta + 2.0, min=eps))

    # standard normal CDF via erf (keeps device/dtype)
    z  = mu_delta / denom
    p_i_t = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    p_i = float(p_i_t.item())
    return p_i, 1.0 - p_i

# --- Pairwise BO with qEUBO (MC), sequential batching, balanced & novel A/Bs ---

import warnings
import math
import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import qExpectedUtilityOfBestOption
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

warnings.filterwarnings("ignore")

dtype = torch.float64
torch.manual_seed(0)
np.random.seed(0)

# ------------------------ Probability helper (probit) ------------------------
@torch.no_grad()
def predict_pref_proba(model, x_i: torch.Tensor, x_j: torch.Tensor):
    """
    Closed-form probit integration (i ~ GP, ε ~ N(0,1) i.i.d.):
    P(i > j) = Φ( (μ_i - μ_j) / sqrt(var_i + var_j - 2*cov_ij + 2) )
    Returns (p_i, p_j) as Python floats.
    """
    X = torch.stack((x_i, x_j), dim=0)  # (2, d)
    post = model.posterior(X)
    m = post.mean.reshape(-1)            # (2,)
    K = post.mvn.covariance_matrix       # (2, 2)

    mu_delta  = m[0] - m[1]
    var_delta = K[0, 0] + K[1, 1] - 2 * K[0, 1]
    eps = torch.finfo(m.dtype).eps
    denom = torch.sqrt(torch.clamp(var_delta + 2.0, min=eps))  # +2 from ε1-ε2
    z  = mu_delta / denom
    p_i_t = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    p_i = float(p_i_t.item())
    return p_i, 1.0 - p_i

# ------------------------ Problem setup ------------------------
# Features: [p_win, win_amount, loss_amount]
# p_win ∈ [0.05, 0.95], amounts ∈ [1, 1000]
dim = 3
bounds = torch.tensor(
    [[0.05,  1.0,   1.0],
     [0.95, 1000.0, 1000.0]],
    dtype=dtype,
)

# Config
NUM_QUERIES  = 20
NUM_RESTARTS = 12
RAW_SAMPLES  = 1024

# Storage
X = torch.empty(0, dim, dtype=dtype)
comps = torch.empty(0, 2, dtype=torch.long)  # [winner, loser]
best_hist = []

def _fmt_gamble(g: torch.Tensor) -> str:
    p, w, l = g[0].item(), g[1].item(), g[2].item()
    return f"{p*100:.0f}% chance of +${w:.0f}, {(1-p)*100:.0f}% chance of -${l:.0f}"

def _ask_ab(prompt: str) -> str:
    while True:
        s = input(prompt).strip().upper()
        if s in ("A", "B"):
            return s
        print("Please enter A or B.")

# ------------------------ Guard + TR helpers ------------------------
def _pairwise_linf(x_new: torch.Tensor, X_seen: torch.Tensor) -> torch.Tensor:
    """Pairwise L_inf distances between x_new (n_new, d) and X_seen (n_seen, d)."""
    if X_seen.numel() == 0:
        return torch.full((x_new.shape[0], 0), float("inf"), dtype=x_new.dtype, device=x_new.device)
    diffs = (x_new[:, None, :] - X_seen[None, :, :]).abs().max(dim=-1).values
    return diffs

def _min_dist_to_seen(x_new: torch.Tensor, X_seen: torch.Tensor) -> torch.Tensor:
    """Min L_inf distance from each row in x_new to any row in X_seen."""
    D = _pairwise_linf(x_new, X_seen)
    if D.numel() == 0:
        return torch.full((x_new.shape[0],), float("inf"), dtype=x_new.dtype, device=x_new.device)
    return D.min(dim=1).values

def _local_bounds(center: torch.Tensor, frac: float = 0.25):
    """Trust region box around `center` (original units)."""
    span = bounds[1] - bounds[0]
    lb = torch.max(bounds[0], center - frac * span)
    ub = torch.min(bounds[1], center + frac * span)
    return torch.stack([lb, ub])

def _ensure_novel_and_nontrivial(
    model,
    pair,                # (2, d)
    X_seen,              # (n_seen, d)
    bounds_,
    acq,
    p_lo=0.40, p_hi=0.60,
    min_linf_frac=0.02,
    max_tries=8,
    num_restarts=5,
    raw_samples=256,
):
    """
    Validates a candidate pair; if too trivial or too close to prior points (or to each other),
    re-optimizes with broader search and refines the second point in a local TR around the first.
    Returns a (2,d) pair.
    """
    span = (bounds_[1] - bounds_[0]).to(pair)
    linf_thresh = (min_linf_frac * span).max()  # scalar threshold in original units

    def _ok(p):
        # far enough from all seen
        md = _min_dist_to_seen(p, X_seen)
        if not (md > linf_thresh).all():
            return False
        # A and B not nearly identical
        ab_dist = (p[0] - p[1]).abs().max()
        if not (ab_dist > linf_thresh):
            return False
        # balanced choice prob
        pA, _ = predict_pref_proba(model, p[0], p[1])
        return (p_lo <= pA <= p_hi)

    if _ok(pair):
        return pair

    # Retry with stronger exploration + TR refinement for the second point
    for _ in range(max_tries):
        # global search with more raw samples
        pair, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_,
            q=2,
            sequential=True,                 # pick 2nd conditional on 1st
            num_restarts=num_restarts,
            raw_samples=raw_samples * 4,
        )

        # Refine x2 in a local TR around x1 to find a fair challenger
        x1 = pair[0].detach()
        tr = _local_bounds(x1, frac=0.25)
        pair_tr, _ = optimize_acqf(
            acq_function=acq,
            bounds=tr,
            q=2,
            sequential=True,
            num_restarts=num_restarts,
            raw_samples=raw_samples * 2,
        )
        pair = torch.stack([x1, pair_tr[1].detach()], dim=0)

        if _ok(pair):
            return pair

    return pair  # best we have if all retries fail

# ------------------------ Initialization ------------------------
init = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(2, dim, dtype=dtype)
print("Initial comparison:")
print("A:", _fmt_gamble(init[0]))
print("B:", _fmt_gamble(init[1]))
choice = _ask_ab("Pick one (A/B): ")

X = torch.cat([X, init], dim=0)
first_pair = torch.tensor([[0, 1]] if choice == "A" else [[1, 0]], dtype=torch.long)
comps = torch.cat([comps, first_pair], dim=0)

# Fit initial model
model = PairwiseGP(X, comps, input_transform=Normalize(d=dim, bounds=bounds))
mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
with torch.no_grad():
    mu = model.posterior(X).mean.view(-1)
best_hist.append(mu.max().item())

# MC sampler (robust across BoTorch versions)
try:
    sampler = SobolQMCNormalSampler(num_samples=128)
except TypeError:
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

# ------------------------ Interactive loop ------------------------
for t in range(1, NUM_QUERIES + 1):
    acq = qExpectedUtilityOfBestOption(pref_model=model, sampler=sampler)

    # Greedy/sequential batch builds a conditional second point
    pair, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=2,
        sequential=True,              # key for better pairs
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    # Guards: ensure novelty & non-triviality; retry + TR refine if needed
    next_X = _ensure_novel_and_nontrivial(
        model=model,
        pair=pair,
        X_seen=X,
        bounds_=bounds,
        acq=acq,
        p_lo=0.40, p_hi=0.60,
        min_linf_frac=0.02,
        max_tries=8,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    print(f"\nQuery {t}/{NUM_QUERIES}:")
    print("A:", _fmt_gamble(next_X[0]))
    print("B:", _fmt_gamble(next_X[1]))

    pA, pB = predict_pref_proba(model, next_X[0].to(dtype), next_X[1].to(dtype))
    print(f"Model-estimated P(choose A) = {pA:.2%}, P(choose B) = {pB:.2%}")

    choice = _ask_ab("Pick A or B: ")

    # Bookkeeping
    n = X.shape[0]
    X = torch.cat([X, next_X.to(dtype)], dim=0)
    pair_lbl = torch.tensor([[n, n + 1]] if choice == "A" else [[n + 1, n]], dtype=torch.long)
    comps = torch.cat([comps, pair_lbl], dim=0)

    # Refit model
    model = PairwiseGP(X, comps, input_transform=Normalize(d=dim, bounds=bounds))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    with torch.no_grad():
        mu = model.posterior(X).mean.view(-1)
        best_hist.append(mu.max().item())

print("\nDone. `X` holds all gambles, `comps` your preferences, `best_hist` logs progress.")
