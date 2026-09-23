# State Estimation in Electric Grids: Dynamic and Learning-Based Algorithms

Classical WLS treats each estimation snapshot independently. Two more recent families exploit temporal structure and data-driven priors instead.

## Kalman Filter-Based Dynamic State Estimation (DSE)

With PMUs providing phasor measurements at 30–60+ Hz, the grid state can be tracked as a stochastic process rather than re-solved from scratch each snapshot:

**xₖ = f(xₖ₋₁) + wₖ**   (process model — often a random-walk or AR model on V, θ)
**zₖ = h(xₖ) + vₖ**   (measurement model, as in WLS)

The Extended Kalman Filter (EKF) linearizes h(·) at each step — via the same Jacobian H used in WLS — and propagates a state covariance Pₖ through predict/update cycles, giving a recursive, cheap estimator that naturally fuses SCADA and PMU data arriving at different rates. The Unscented Kalman Filter (UKF) avoids explicit Jacobian linearization by propagating a deterministic sigma-point set through the true nonlinear h(·), improving accuracy when the system is strongly nonlinear (e.g. near voltage collapse or during large disturbances) at moderate extra cost. Both inherit WLS's Gaussian-noise assumptions and need a reasonably accurate process model; a poor choice of f(·) or process noise covariance Q can cause filter divergence, particularly after topology changes.

## Learning-Based and Latent-Space Approaches

A more recent direction replaces or augments h(·) and the estimator itself with learned models. Two threads stand out:

- **Deep learning regressors** — neural networks (GNNs are a natural fit, since the grid graph maps directly onto message-passing structure) trained to map measurements directly to x, either supervised on historical/simulated (measurement, state) pairs or as a learned correction layered on a physics-based estimate. GNNs exploit the same sparsity structure as H and can generalize across topologies better than a fixed-Jacobian method.

- **Latent-space / self-supervised estimation** — architectures such as JEPA-style predictors learn a compressed representation of plausible grid states from unlabeled data, exploiting the fact that true states lie on a low-dimensional manifold shaped by the power-flow constraints (the same manifold geometry underlying classical PF Jacobian analysis). Estimation then becomes inference in that latent space rather than the full AC state space — improving robustness to missing or corrupted measurements and reducing effective problem dimensionality, at the cost of needing representative training data and losing some of the interpretability and hard convergence guarantees of iterative physics-based solvers.

These families are not mutually exclusive: several current approaches use learned models to initialize or regularize a classical WLS/Kalman step, combining data-driven priors with the residual guarantees of the physics-based formulation.
