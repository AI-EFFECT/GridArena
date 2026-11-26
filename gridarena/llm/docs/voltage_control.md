# Voltage Agent

## Overview

The Voltage Agent provides the Voltage Control Benchmark. The goal is to propose corrective actions — adjustments to active power, reactive power, or voltage setpoints — that bring voltages back within acceptable limits when violations are detected.

**Voltage violation types:**
- **Overvoltage** (above 1.1 pu / 253V): typically caused by excess solar PV generation injecting power back into the grid
- **Undervoltage** (below 0.9 pu / 207V): typically caused by heavy loads such as EV charging or industrial demand

This is an optimisation problem: find the minimum corrective intervention that resolves all violations. The benchmark provides training scenarios with reference solutions. The test snapshots are never downloaded: instead you submit a Python script defining `predict(snapshot)`, the server runs it on the hidden test snapshots and scores its corrective actions.

---

## Functions

### get_training_voltage_control_data

Use this when you want to develop or train a voltage control algorithm, understand what reference solutions look like, or study how specific violations are corrected in practice.

**Parameters:**
- `voltage_limit` (float, required): The voltage threshold (in volts) used to define a violation, e.g. `230.0`
- `scenario` (string, optional, default=`"Overvoltages"`): The type of violation to train on. Use `"Overvoltages"` for solar-driven high voltage, `"Undervoltages"` for load-driven low voltage.
- `difficulty` (string, optional, default=`"clean"`): Controls measurement noise level

**Returns:** Scenarios including voltage measurements, violation details, and reference corrective actions.

**How to interpret:**
- Each scenario shows a snapshot of the grid under stress with one or more nodes outside the acceptable voltage range
- Reference actions show what power adjustments were applied to resolve each violation
- Use these to train or validate your control strategy before running on test data

**Example:**
```json
{
  "voltage_limit": 230.0,
  "scenario": "Overvoltages",
  "difficulty": "clean"
}
```

---

### get_voltage_control_submission_template

Use this when you want to write a voltage control method to submit, or need to know the exact input and output format the submitted script must follow.

**Parameters:** none.

**Returns:** `filename` and `source` of the blank submission script.

**How to interpret:**
- The script defines `predict(snapshot)`, called once per test snapshot (a grid at a timestamp with at least one violation)
- `snapshot` has `datetime`, `scenario`, `voltage_limit`, `nominal_voltage` (230 V), `voltage_band` (0.05) and `measurements` (a list of `node_id`, `phase`, `power_active`, `power_reactive`, `voltage_magnitude`, `voltage_angle`)
- It must return a list of actions `{node_id, phase, corrected_voltage, adjusted_power_active, adjusted_power_reactive}`, the format of the training reference solutions; a node it leaves out counts as unchanged
- The template's default `predict` is a simple baseline; replace its body with the actual method
- The script runs without network access, with numpy, pandas, scipy, networkx and torch available

---

### submit_voltage_control_script

Use this when you want to evaluate a voltage control method on the hidden test data.

**Parameters:**
- `submission_name` (string, required): A label for this attempt (e.g. `"run_001"`); letters, digits, `_`, `-`, `.`
- `code` (string, required): The full Python source, based on the submission template
- `voltage_limit` (float, required): Voltage threshold (V) that selects the snapshots with violations, e.g. `241.5`
- `scenario` (string, optional, default=`"Overvoltages"`): `"Overvoltages"` or `"Undervoltages"`
- `difficulty` (string, optional, default=`"clean"`): Filters the snapshots by hardness

**Returns:** The submission record with its `submission_id` and `status` (`queued`).

**Notes:**
- The script is checked for syntax errors and for a top-level `predict` function before it is queued
- Evaluation runs in the background; use `get_voltage_control_submission` with the returned `submission_id` to follow it
- The test data itself is never downloadable: only the submitted script sees it

**Example:**
```json
{
  "submission_name": "run_001",
  "code": "def predict(snapshot):\n    return [{'node_id': m['node_id'], 'phase': m['phase'], 'corrected_voltage': m['voltage_magnitude']} for m in snapshot['measurements']]\n",
  "voltage_limit": 241.5,
  "scenario": "Overvoltages",
  "difficulty": "clean"
}
```

---

### get_voltage_control_submission

Use this when you want to know whether a submitted script has finished and how well it did.

**Parameters:**
- `submission_id` (string, required): The ID returned by `submit_voltage_control_script`

**Returns:** The submission's `status` (`queued`, `running`, `completed`, `failed`), its `score` when completed, its `error` when failed, and `log_tail` (what the script printed, plus tracebacks).

**How to interpret:**
- `score.score` is between 0 and 1, higher is better: the node-weighted mean of the snapshot scores
- A snapshot with reference solutions rewards matching the reference corrected voltages (within 1 V), then the least total change in active power; without a reference it rewards moving voltages back towards nominal
- `score.per_snapshot` gives each snapshot's score, scoring mode, reference-voltage error and effort; `score.per_grid` aggregates per test grid
- When `status` is `failed`, read `error` and `log_tail` to fix the script and submit again

**Example:**
```json
{
  "submission_id": "3f9c1a2b7d4e"
}
```
