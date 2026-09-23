# Phase Detection in LV Electric Grids: Signal-Based and Graph/ML Methods

## Signal Injection (Pilot-Tone / PLC-Based)

The physical alternative to data-driven inference: inject a known signal onto one phase at the substation or transformer and detect its presence at the customer meter. Classical implementations use a low-frequency pilot tone (in the same family as ripple/tariff-control signaling) or a dedicated PLC probe signal impressed on a single phase; a receiver — either a portable clamp-on device used by a field technician or a capability built into the smart meter itself — reports which phase carries the signal, directly yielding φᵢ with no statistical inference step.

This class of method is largely load- and topology-independent and gives near-deterministic accuracy, which is why it remains the practical fallback DSOs use to *validate* results from data-driven PI. Its drawback is operational cost: it needs injection hardware at the source, per-customer measurement (or AMI meters with PLC receive capability), and — for portable field methods — technician time proportional to feeder size, making it impractical as the primary method for city-scale relabeling campaigns.

## Graph-Constrained and Learning-Based Methods

A separate family treats PI as inference over the network graph rather than a per-customer independent decision, since phase assignment is coupled with grid topology (itself often partially unknown in LV networks) and must respect physical constraints, e.g. all customers downstream of a given single-phase service line share one phase.

- **Joint topology–phase estimation**: formulated as a combinatorial or mixed-integer optimization (or relaxed to a convex/graph-regularized problem) that jointly infers the radial connectivity tree and each node's phase label, using smart-meter voltage/power measurements as noisy evidence and the physical network constraints as hard structure. This tends to outperform independent per-customer correlation when topology is uncertain, at higher computational cost.
- **Semi-supervised graph learning**: label propagation or graph neural networks that use a sparse set of verified anchor labels plus voltage-correlation-derived edge weights between meters, propagating phase labels across the graph rather than trusting each node's correlation score in isolation — helpful precisely in the weak-signal, well-balanced-feeder regime where plain correlation degrades.
- **Supervised classifiers**: trained on synthetic or historical labeled data, taking engineered features (correlation to each reference phase, voltage variance, load profile shape statistics) or raw time series as input; GNNs are a natural fit here too, mirroring their use in state estimation, since the LV feeder graph structure can be encoded directly into the model.

In practice these graph/ML methods are usually deployed as a refinement layer on top of correlation or clustering outputs — resolving the ambiguous cases (high pairwise phase correlation, sparse verified anchors) rather than replacing the cheaper baseline everywhere.
