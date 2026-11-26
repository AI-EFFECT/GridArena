# State Agent

## Overview

The State Agent provides the State Estimation Benchmark. The goal is to predict missing voltage values (magnitude and angle) at nodes that are not fully observed, using the measurements that are available and knowledge of how power flows through the network.

This is a regression problem grounded in power system physics: voltage at unobserved nodes is not random — it is constrained by Kirchhoff's laws and the grid topology. Good algorithms exploit this structure.

The benchmark provides training data with every series measured. The test tasks are never downloaded: instead you submit a Python script defining `predict(task)`, the server runs it on hidden estimation tasks (a few timestamps per test grid) and scores the voltages it estimates.

**Typical use case:** A grid operator has smart meters at some nodes but not all. State estimation fills in the missing voltage picture across the full network.

---

## Functions

### get_training_state_data

Use this when you want to develop or train a state estimation algorithm, access fully observable data where all node voltages are known, or understand what the input/output relationship looks like before attempting predictions.

**Parameters:**
- `noise_difficulty` (string, optional, default=`"clean"`): Controls measurement noise. `"clean"` provides ideal readings; `"noisy"` simulates realistic sensor noise.
- `observability` (string, optional, default=`"medium"`): Controls how many nodes have visible measurements. `"low"` means fewer observed nodes; `"high"` means most nodes are observed.

**Returns:** Full time-series measurements for all nodes including voltage magnitude and angle — no masking.

**How to interpret:**
- Use this to learn the relationship between observed nodes and unobserved nodes
- Voltage magnitude is typically around 230V; significant deviations indicate load stress or generation
- Voltage angle propagates through the network according to power injection and impedance

**Example:**
```json
{
  "noise_difficulty": "clean",
  "observability": "medium"
}
```

---

### get_state_estimation_submission_template

Use this when you want to write a state estimation method to submit, or need to know the exact input and output format the submitted script must follow.

**Parameters:** none.

**Returns:** `filename` and `source` of the blank submission script.

**How to interpret:**
- The script defines `predict(task)`, called once per estimation task (one test grid at one timestamp)
- `task` has `timestamp`, `known` (series key -> measurement at the timestamp), `unknown` (series keys to estimate) and `history` (series key -> records before the timestamp; unknown series stop a few hours earlier)
- It must return `{series_key: {"voltage_magnitude": ..., "voltage_angle": ...}}` for every unknown key
- The template's default `predict` is a simple baseline; replace its body with the actual method
- The script runs without network access, with numpy, pandas, scipy, networkx and torch available

---

### submit_state_estimation_script

Use this when you want to evaluate a state estimation method on the hidden test data.

**Parameters:**
- `submission_name` (string, required): A label for this attempt (e.g. `"run_001"`); letters, digits, `_`, `-`, `.`
- `code` (string, required): The full Python source, based on the submission template
- `noise_difficulty` (string, optional, default=`"clean"`): Noise level of the test measurements
- `observability` (string, optional, default=`"medium"`): How many series are unknown and how much of their history is hidden

**Returns:** The submission record with its `submission_id` and `status` (`queued`).

**Notes:**
- The script is checked for syntax errors and for a top-level `predict` function before it is queued
- Evaluation runs in the background; use `get_state_estimation_submission` with the returned `submission_id` to follow it
- The test data itself is never downloadable: only the submitted script sees it

**Example:**
```json
{
  "submission_name": "run_001",
  "code": "def predict(task):\n    v = [m['voltage_magnitude'] for m in task['known'].values()]\n    avg = sum(v) / len(v) if v else 230.0\n    return {k: {'voltage_magnitude': avg, 'voltage_angle': 0.0} for k in task['unknown']}\n",
  "noise_difficulty": "clean",
  "observability": "medium"
}
```

---

### get_state_estimation_submission

Use this when you want to know whether a submitted script has finished and how well it did.

**Parameters:**
- `submission_id` (string, required): The ID returned by `submit_state_estimation_script`

**Returns:** The submission's `status` (`queued`, `running`, `completed`, `failed`), its `score` when completed, its `error` when failed, and `log_tail` (what the script printed, plus tracebacks).

**How to interpret:**
- `score.score` = mean squared voltage-magnitude error + mean |sin| of the angle error, over every unknown series of every task: lower is better
- A missing or non-numeric estimate is scored as 0 V with the worst angle, so always return one
- `score.rmse_voltage_magnitude` and `score.mae_voltage_angle` are over the usable estimates; `score.per_grid` breaks results down per test grid
- Every submission with the same settings gets exactly the same tasks, so scores are comparable
- When `status` is `failed`, read `error` and `log_tail` to fix the script and submit again

**Example:**
```json
{
  "submission_id": "3f9c1a2b7d4e"
}
```
