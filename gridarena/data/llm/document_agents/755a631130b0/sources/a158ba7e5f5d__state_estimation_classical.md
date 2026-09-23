# State Estimation in Electric Grids: Classical Algorithms

Power system state estimation (SE) recovers the vector of bus voltage magnitudes and angles **x = (V, θ)** that best explains a redundant set of noisy measurements **z** (power flows, injections, voltage magnitudes, and increasingly PMU phasors), given the nonlinear measurement model:

**z = h(x) + e**

where h(·) encodes the AC power flow equations through the network admittance matrix Y, and e is measurement noise.

## Weighted Least Squares (WLS)

The dominant industrial algorithm. Assuming Gaussian, independent measurement errors with covariance R = diag(σᵢ²), WLS minimizes:

**J(x) = [z − h(x)]ᵀ R⁻¹ [z − h(x)]**

Solved iteratively via Gauss–Newton: at each step, linearize h around the current estimate using the measurement Jacobian H = ∂h/∂x, and solve the normal equations:

**(HᵀR⁻¹H) Δx = HᵀR⁻¹[z − h(x)]**

The gain matrix G = HᵀR⁻¹H is sparse and symmetric positive definite (given observability), typically factorized via sparse Cholesky or QR decomposition — plain normal equations can be ill-conditioned when the measurement set mixes very different scales (e.g. voltage magnitudes vs. power injections).

WLS is efficient and statistically optimal under its Gaussian assumption, but sensitive to bad data: a single gross outlier biases the whole estimate, since squared residuals give unbounded influence to large errors. Post-estimation bad-data detection (χ² test on J(x), followed by largest-normalized-residual search) is the standard remedy — a separate, iterative bolt-on rather than an intrinsic robustness property.

## Robust Estimators: WLAV and M-Estimators

Weighted Least Absolute Value (WLAV) replaces the quadratic objective with an L1 norm:

**J(x) = Σᵢ wᵢ |zᵢ − hᵢ(x)|**

L1 minimization has a bounded influence function — a single bad measurement affects the estimate far less than under WLS — at the cost of solving a linear program (or iteratively reweighted least squares, IRLS) at each Gauss–Newton step instead of a single linear solve. WLAV has a known weakness, though: it can fail to reject *leverage points* (outliers with high influence on H itself, e.g. at poorly-connected buses), which motivates projection-based or Schweppe-type generalizations.

More general M-estimators (Huber, Welsch, Schweppe-Huber generalized-M) interpolate between L2 efficiency near the bulk of the data and L1-like robustness in the tails, via a bounded-derivative loss ψ(·) applied to normalized residuals. They retain most of WLS's statistical efficiency under nominal Gaussian noise while capping outlier influence — a common choice where SCADA/PMU data quality is inconsistent.
