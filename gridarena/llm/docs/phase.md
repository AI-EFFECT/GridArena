# Phase Agent

## Overview

The Phase Agent provides the Phase Identification Benchmark. The goal is to determine which phase (R, S, or T) each node in the grid belongs to, using only smart meter time-series data.

This is a classification problem: each measurement series must be assigned to exactly one of three phases. The benchmark provides labelled training data (with known phases). The test data is never downloaded: instead you submit a Python script defining `predict(measurements)`, the server runs it on the hidden test grids and scores the phases it returns.

**Typical use case:** An engineer or algorithm wants to identify the phase assignment of nodes in a grid where the wiring is poorly documented.

---

## Functions

### get_training_phase_data

Use this when you want to train or test a phase identification algorithm, get labelled examples of what phase each node belongs to, or understand what the input data looks like before attempting predictions.

**Parameters:**
- `difficulty` (string, optional, default=`"clean"`): Controls the noise level in the measurements. One of `"clean"` (noise-free), `"easy"`, `"medium"`, `"hard"` (increasing sensor noise and outliers).

**Returns:** Time-series measurements per node with known phase labels (R, S, or T) included.

**How to interpret:**
- Each record contains node measurements and the correct phase label
- Use this data to train a classifier or develop heuristics
- Nodes on the same phase tend to have correlated power and voltage patterns

**Example:**
```json
{
  "difficulty": "clean"
}
```

---

### get_phase_submission_template

Use this when you want to write a phase identification model to submit, or need to know the exact input and output format the submitted script must follow.

**Parameters:** none.

**Returns:** `filename` and `source` of the blank submission script.

**How to interpret:**
- The script defines `predict(measurements)`, called once per test grid
- `measurements` maps a random series key to its list of records (`datetime`, `power_active`, `power_reactive`, `voltage_magnitude`, `voltage_angle`), the same fields as the training data without `phase`
- It must return `{series_key: "R" | "S" | "T"}` for every key
- The template's default `predict` guesses at random; replace its body with the actual method
- The script runs without network access, with numpy, pandas, scipy and torch available

---

### submit_phase_script

Use this when you want to evaluate a phase identification method on the hidden test grids.

**Parameters:**
- `submission_name` (string, required): A label for this attempt (e.g. `"run_001"`); letters, digits, `_`, `-`, `.`
- `code` (string, required): The full Python source, based on the submission template
- `difficulty` (string, optional, default=`"clean"`): Noise level of the test data the script is run on

**Returns:** The submission record with its `submission_id` and `status` (`queued`).

**Notes:**
- The script is checked for syntax errors and for a top-level `predict` function before it is queued
- Evaluation runs in the background; use `get_phase_submission` with the returned `submission_id` to follow it

**Example:**
```json
{
  "submission_name": "run_001",
  "code": "def predict(measurements):
    return {key: 'R' for key in measurements}
",
  "difficulty": "clean"
}
```

---

### get_phase_submission

Use this when you want to know whether a submitted script has finished and how accurate it was.

**Parameters:**
- `submission_id` (string, required): The ID returned by `submit_phase_script`

**Returns:** The submission's `status` (`queued`, `running`, `completed`, `failed`), its `score` when completed, its `error` when failed, and `log_tail` (what the script printed, plus tracebacks).

**How to interpret:**
- `score.accuracy` = correctly predicted series / all test series; missing keys and values other than R/S/T count as wrong
- `score.per_grid` lists each test grid's accuracy and any error its `predict` call raised
- `score.confusion` counts true phase vs. predicted phase (`other` = missing or invalid)
- Random guessing gives approximately 0.33
- When `status` is `failed`, read `error` and `log_tail` to fix the script and submit again

**Example:**
```json
{
  "submission_id": "3f9c1a2b7d4e"
}
```
