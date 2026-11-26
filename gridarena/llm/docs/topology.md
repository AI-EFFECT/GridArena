# Topology Agent

## Overview

The Topology Agent provides the Topology Discovery Benchmark. The goal is to reconstruct the physical electrical connections (edges) between nodes using only smart meter time-series data — without access to the actual wiring diagram.

This is a graph reconstruction problem: given measurements at each node, infer which nodes are directly connected by a cable. The benchmark provides labelled training data (with known connections). The test grids are never downloaded: instead you submit a Python script defining `predict(grid)`, the server runs it on the hidden test grids and scores the edges it returns.

**Typical use case:** An engineer wants to verify or recover the network topology of a grid where wiring documentation is incomplete or unreliable.

---

## Functions

### get_training_topology_data

Use this when you want to train a topology discovery algorithm, see examples of what the correct connection graph looks like, or understand the relationship between measurements and physical connections.

**Parameters:**
- `difficulty` (string, optional, default=`"clean"`): Controls measurement noise. `"clean"` provides ideal data; `"noisy"` simulates realistic smart meter noise.

**Returns:** Time-series measurements per node along with the correct edge list (the true topology).

**How to interpret:**
- Each edge represents a physical cable between two nodes
- Nodes that are directly connected tend to have strongly correlated voltage and power patterns
- The reference node "PT" (transformer) is always connected to at least one other node

**Example:**
```json
{
  "difficulty": "clean"
}
```

---

### get_topology_submission_template

Use this when you want to write a topology discovery method to submit, or need to know the exact input and output format the submitted script must follow.

**Parameters:** none.

**Returns:** `filename` and `source` of the blank submission script.

**How to interpret:**
- The script defines `predict(grid)`, called once per test grid
- `grid` has `nodes` (every node key, also nodes without meters), `root` (the substation key, or null) and `measurements` (node key -> records with `datetime`, `phase`, `power_active`, `power_reactive`, `voltage_magnitude`, `voltage_angle`)
- Node keys are random per submission, so they carry no hint of the wiring
- It must return the grid's edges as a list of `[node_key, node_key]` pairs (undirected)
- The template's default `predict` is a simple baseline; replace its body with the actual method
- The script runs without network access, with numpy, pandas, scipy, networkx and torch available

---

### submit_topology_script

Use this when you want to evaluate a topology discovery method on the hidden test data.

**Parameters:**
- `submission_name` (string, required): A label for this attempt (e.g. `"run_001"`); letters, digits, `_`, `-`, `.`
- `code` (string, required): The full Python source, based on the submission template
- `difficulty` (string, optional, default=`"clean"`): Noise level of the test measurements

**Returns:** The submission record with its `submission_id` and `status` (`queued`).

**Notes:**
- The script is checked for syntax errors and for a top-level `predict` function before it is queued
- Evaluation runs in the background; use `get_topology_submission` with the returned `submission_id` to follow it
- The test data itself is never downloadable: only the submitted script sees it

**Example:**
```json
{
  "submission_name": "run_001",
  "code": "def predict(grid):\n    return [[grid['root'], n] for n in grid['nodes'] if n != grid['root']]\n",
  "difficulty": "clean"
}
```

---

### get_topology_submission

Use this when you want to know whether a submitted script has finished and how well it did.

**Parameters:**
- `submission_id` (string, required): The ID returned by `submit_topology_script`

**Returns:** The submission's `status` (`queued`, `running`, `completed`, `failed`), its `score` when completed, its `error` when failed, and `log_tail` (what the script printed, plus tracebacks).

**How to interpret:**
- `score.f1` is the headline metric: the F1 of precision (correct / predicted edges) and recall (correct / true edges), over all test grids
- Returning every possible pair gives full recall but very low precision, so F1 penalises guessing
- `score.per_grid` lists each test grid's edges found, wrong edges, invalid pairs and any error its `predict` call raised
- When `status` is `failed`, read `error` and `log_tail` to fix the script and submit again

**Example:**
```json
{
  "submission_id": "3f9c1a2b7d4e"
}
```
