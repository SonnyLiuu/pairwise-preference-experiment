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

# Features: [p_win, win_amount, loss_amount]
# p_win ∈ [0.05, 0.95], amounts ∈ [1, 1000]
dim = 3
bounds = torch.tensor(
    [[0.05,  1.0,   1.0],
     [0.95, 1000.0, 1000.0]],
    dtype=dtype,
)

# CONFIG
NUM_QUERIES  = 20
NUM_RESTARTS = 5
RAW_SAMPLES  = 256
BALD_SAMPLES = 256

# storage
X = torch.empty(0, dim, dtype=dtype)
comps = torch.empty(0, 2,  dtype=torch.long)  # [winner, loser] pairs
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

# give a initializing A/B choice
init = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(2, dim, dtype=dtype)
print("Initial comparison:")
print("A:", _fmt_gamble(init[0]))
print("B:", _fmt_gamble(init[1]))
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
    next_X, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=2,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    print(f"\nQuery {t}/{NUM_QUERIES}:")
    print("A:", _fmt_gamble(next_X[0]))
    print("B:", _fmt_gamble(next_X[1]))

    # calculate and print model's current prediction of user choice
    pA, pB = predict_pref_proba(model, next_X[0].to(dtype), next_X[1].to(dtype))
    print(f"Model-estimated P(choose A) = {pA:.2%}, P(choose B) = {pB:.2%}")

    choice = _ask_ab("Pick A or B: ")

    # add these two gambles and the comparison
    n = X.shape[0]
    X = torch.cat([X, next_X.to(dtype)], dim=0)
    pair = torch.tensor([[n, n + 1]] if choice == "A" else [[n + 1, n]], dtype=torch.long)
    comps = torch.cat([comps, pair], dim=0)

    # refit and log best predicted among observed gambles
    model = PairwiseGP(X, comps, input_transform=Normalize(d=dim, bounds=bounds))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    with torch.no_grad():
        mu = model.posterior(X).mean.view(-1)
        best_hist.append(mu.max().item())

print("\nDone. `X` holds all gambles, `comps` your preferences, `best_hist` logs progress.")

def sample_probe_pairs(n_pairs: int, bounds: torch.Tensor, seed: int = 123):
    """
    Returns list of (xA, xB) tensors, never used for training.
    """
    g = torch.Generator().manual_seed(seed)
    low, high = bounds[0], bounds[1]
    pairs = []
    for _ in range(n_pairs):
        xA = low + (high - low) * torch.rand(bounds.shape[1], generator=g, dtype=dtype)
        xB = low + (high - low) * torch.rand(bounds.shape[1], generator=g, dtype=dtype)
        pairs.append((xA, xB))
    return pairs

def evaluate_on_pairs(model, pairs, labels):
    """
    labels: 1 if A chosen, 0 if B chosen
    """
    pA_list, y_list = [], []
    for (xA, xB), y in zip(pairs, labels):
        pA, _ = predict_pref_proba(model, xA, xB)
        pA_list.append(float(pA))
        y_list.append(int(y))

    pA = np.asarray(pA_list, dtype=float)
    y  = np.asarray(y_list, dtype=int)
    acc = float(np.mean((pA >= 0.5) == (y == 1)))
    p_true = np.where(y == 1, pA, 1.0 - pA)
    logloss = float(-np.mean(np.log(p_true + 1e-12)))
    brier = float(np.mean((pA - y)**2))
    return {"n": int(len(y)), "accuracy": acc, "log_loss": logloss, "brier": brier}

# Test-Set Config
N_PROBES   = 10  # how many test gambles to generate
PROBE_SEED = 123

probe_pairs = sample_probe_pairs(N_PROBES, bounds, seed=PROBE_SEED)
probe_labels = [None] * N_PROBES          # 1 for 'A', 0 for 'B'
probe_probabilities = [None] * N_PROBES   # aligned with probe_pairs

print("\n--- HELD-OUT TEST SET (not used for training) ---")
for idx, (xA, xB) in enumerate(probe_pairs, start=1):
    print(f"\n[TEST {idx}/{N_PROBES}]")
    print("A:", _fmt_gamble(xA))
    print("B:", _fmt_gamble(xB))

    # Prediction BEFORE seeing the user's answer
    pA, pB = predict_pref_proba(model, xA, xB)
    msg = f"Model-estimated P(choose A) = {pA:.2%}, P(choose B) = {pB:.2%}"
    probe_probabilities[idx - 1] = msg

    while True:
        choice = input("Pick one (A/B): ").strip().upper()
        if choice in ("A", "B"):
            probe_labels[idx - 1] = 1 if choice == "A" else 0
            break
        print("Please enter A or B.")

print("\nTesting Set Complete!")

answered_idx = [i for i, y in enumerate(probe_labels) if y is not None]

if not answered_idx:
    print("\nNo test answers collected; nothing to evaluate.")
else:
    for k, i in enumerate(answered_idx, 1):
        xA, xB = probe_pairs[i]
        y = probe_labels[i]
        print(f"\n[TEST {k}/{len(answered_idx)}]")
        print("A:", _fmt_gamble(xA))
        print("B:", _fmt_gamble(xB))
        print(probe_probabilities[i])
        print(f"User Choice: {'A' if int(y) == 1 else 'B'}")

    eval_pairs  = [probe_pairs[i]  for i in answered_idx]
    eval_labels = [probe_labels[i] for i in answered_idx]
    scores = evaluate_on_pairs(model, eval_pairs, eval_labels)
    print("\nHELD-OUT TEST performance:", scores)
