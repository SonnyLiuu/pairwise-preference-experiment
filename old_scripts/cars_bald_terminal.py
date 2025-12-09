!pip install torch
!pip install botorch

import torch
import math

# Probability Algorithm
@torch.no_grad()
def predict_pref_proba(model, x_i: torch.Tensor, x_j: torch.Tensor):
    """
    Closed-form probit integration (i ~ GP, ε ~ N(0,1) i.i.d.):
    P(i > j) = Φ( (μ_i - μ_j) / sqrt(var_i + var_j - 2*cov_ij + 2) )
    Returns (p_i, p_j) as Python floats.
    """
    X = torch.stack((x_i, x_j), dim=0)  # (2, d), same device/dtype
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

import warnings
import numpy as np

from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import PairwiseBayesianActiveLearningByDisagreement
from botorch.optim import optimize_acqf

warnings.filterwarnings("ignore")

dtype = torch.float64
torch.manual_seed(0)
np.random.seed(0)

# Car Experiment Setup
# Encode each feature as {0, 1, 2}:
# Brand: 0=Benz, 1=Toyota, 2=Ford
# Color: 0=Black, 1=White, 2=Red
# Fuel:  0=Hybrid, 1=Gas,   2=Electric

attr_names = ["Brand", "Color", "Fuel"]
brands = ["Benz", "Toyota", "Ford"]
colors = ["Black", "White", "Red"]
fuels  = ["Hybrid", "Gas", "Electric"]

# Save user's true preference order
def _ask_ranked_order(attr_name: str, options: list[str]) -> list[str]:
  while True:
    raw = input(
        f"\nRank {'/'.join(options)} from most preferred to least preferred.\n"
        f"Separate each {attr_name.lower()} with a space.\n")
    ranking = [word.capitalize() for word in raw.strip().split()]

    if not ranking or len(ranking) != len(options):
      print(f"Please enter {len(options)} {attr_name}s")
      continue

    # Check for unrecognized inputs
    if any(word not in options for word in ranking):
      issue = [p for p in ranking if p not in options]
      print(f"Unrecognized {attr_name.lower()}: {issue}, please try again.")
      continue

    # Check for duplicates
    if len(set(ranking)) != len(ranking):
      print("Duplicates found. Each option must appear exactly once.")
      continue

    return ranking

true_prefs = {
    "Brand": _ask_ranked_order("Brand", brands),
    "Color": _ask_ranked_order("Color", colors),
    "Fuel":  _ask_ranked_order("Fuel", fuels),
}

print("\nSelf-reported preferences:")
for k, v in true_prefs.items():
  print(f"{k}: " + " -> ".join(v))
print("\n")

# CONFIG
dim = 3
bounds = torch.tensor(
    [[0.0, 0.0, 0.0],
     [2.0, 2.0, 2.0]],
    dtype=dtype,
)

NUM_QUERIES  = 20
NUM_RESTARTS = 5
RAW_SAMPLES  = 256
BALD_SAMPLES = 256

# storage
X = torch.empty(0, dim, dtype=dtype)
comps = torch.empty(0, 2,  dtype=torch.long)  # [winner, loser] pairs
best_hist = []


def _snap_discrete(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest valid {0,1,2} index per dimension."""
    return x.round().clamp(0.0, 2.0)


def _fmt_car(x: torch.Tensor) -> str:
    ix = _snap_discrete(x).to(torch.int64)
    b = brands[ix[0].item()]
    c = colors[ix[1].item()]
    f = fuels[ix[2].item()]
    return f"{c} {b} ({f})"


def _ask_ab(prompt: str) -> str:
    while True:
        s = input(prompt).strip().upper()
        if s in ("A", "B"):
            return s
        print("Please enter A or B.")

# give an initializing A/B choice (sample two random discrete cars)
init_idx = torch.randint(0, 3, (2, dim))
init = init_idx.to(dtype=dtype)

print("Query 0/20:")
print("A:", _fmt_car(init[0]))
print("B:", _fmt_car(init[1]))
choice = _ask_ab("Pick one (A/B): ")

X = torch.cat([X, init], dim=0)
first_pair = torch.tensor([[0, 1]] if choice == "A" else [[1, 0]], dtype=torch.long)
comps = torch.cat([comps, first_pair], dim=0)

# fit initial model on initial choice
model = PairwiseGP(X, comps, input_transform=Normalize(d=dim, bounds=bounds))
mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
with torch.no_grad():
    mu = model.posterior(X).mean.view(-1)
best_hist.append(mu.max().item())

# interaction loop
for t in range(1, NUM_QUERIES + 1):
    acq = PairwiseBayesianActiveLearningByDisagreement(
        pref_model=model,
        num_samples=BALD_SAMPLES,
    )
    next_X_cont, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=2,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    # snap to nearest valid car configurations
    next_X = _snap_discrete(next_X_cont).to(dtype)

    if torch.all(next_X[0] == next_X[1]):
        # pick a random *different* car for B
        levels = torch.arange(3, dtype=dtype)
        all_cars = torch.stack(
            torch.meshgrid(levels, levels, levels, indexing="ij"),
            dim=-1,
        ).view(-1, dim)
        mask = ~(all_cars == next_X[0]).all(dim=1)
        candidates = all_cars[mask]
        j = torch.randint(0, candidates.size(0), (1,))
        next_X[1] = candidates[j]

    print(f"\nQuery {t}/{NUM_QUERIES}:")
    print("A:", _fmt_car(next_X[0]))
    print("B:", _fmt_car(next_X[1]))

    # calculate and print model's current prediction of user choice
    pA, pB = predict_pref_proba(model, next_X[0], next_X[1])
    print(f"Model-estimated P(choose A) = {pA:.2%}, P(choose B) = {pB:.2%}")

    choice = _ask_ab("Pick A or B: ")

    # add these two cars and the comparison
    n = X.shape[0]
    X = torch.cat([X, next_X], dim=0)
    pair = torch.tensor([[n, n + 1]] if choice == "A" else [[n + 1, n]], dtype=torch.long)
    comps = torch.cat([comps, pair], dim=0)

    # refit and log best predicted among observed cars
    model = PairwiseGP(X, comps, input_transform=Normalize(d=dim, bounds=bounds))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    with torch.no_grad():
        mu = model.posterior(X).mean.view(-1)
        best_hist.append(mu.max().item())

print("\nDone.")
# 'X' holds all cars as indices [Brand, Color, Fuel] in {0,1,2}
# 'comps' holds pairwise preferences, and 'best_hist' logs progress.