# Phase Detection in LV Electric Grids: Correlation and Clustering-Based Methods

Phase identification (PI) recovers, for each customer or meter on a low-voltage feeder, which of the three phases (A/B/C) it is physically connected to. This connectivity is frequently missing, incomplete, or wrong in DSO records — a legacy of manual wiring and undocumented reconfigurations — yet it is a prerequisite for unbalanced three-phase state estimation, loss allocation, and DER/EV hosting-capacity studies on the LV network.

## Voltage Correlation Against a Reference

The most widely used data-driven approach exploits the fact that voltage magnitude at a customer's meter tracks the voltage of the phase it is connected to, driven by shared loading and by the LV transformer's per-phase regulation. Given smart-meter voltage magnitude time series Vᵢ(t) for each customer i and reference time series V_A(t), V_B(t), V_C(t) measured at the distribution transformer's LV busbar, the assigned phase is:

**φᵢ = argmax_φ ρ(Vᵢ, V_φ)**

where ρ is the Pearson correlation coefficient (sometimes Spearman, or a distance metric such as Euclidean/DTW on normalized series). The method requires only synchronized voltage magnitude readings — no phase angle — which matches what most AMI smart meters already report at 15–30 min or hourly resolution.

Its main weakness is signal strength: correlation across phases collapses toward 1 on lightly loaded or well-balanced feeders, since all three phases then track the same slow, common-mode voltage drift (e.g. from upstream MV regulation) rather than their own idiosyncratic load variation. Accuracy improves with longer observation windows, higher sampling rates, and periods of high load diversity; some formulations instead correlate voltage *deviations* from a rolling mean to suppress the common-mode component before comparing phases.

## Unsupervised Clustering

When a reliable per-phase reference series is unavailable (e.g. no transformer-level metering), customers can instead be clustered directly on their voltage time series — typically k-means or hierarchical clustering with k = 3 — under the assumption that customers on the same physical phase form a tighter cluster than customers on different phases. Cluster identities are then mapped to actual phase labels (A/B/C) using a small number of anchor meters with known, verified connectivity, or via majority voting against any partial ground truth.

Clustering avoids dependence on a clean reference signal but introduces its own fragility: it assumes the dominant source of voltage variation is phase membership rather than, say, feeder distance or local load type, which can break down on short, lightly-loaded laterals. In practice, correlation-based and clustering-based methods are often combined — clustering to get a coarse partition, correlation against a reference (or against a few verified anchors) to resolve the phase label of each cluster.
