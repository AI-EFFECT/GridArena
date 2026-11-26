# Powerflow Agent

## Overview

The Powerflow Agent runs a power flow on a registered low-voltage grid: from the measured active/reactive power of every node and the measured voltage at the substation node "PT", it computes the complex voltage (magnitude and angle) of every node, on all three phases, for each measurement timestamp. The method is a backward/forward sweep for radial grids.

Two models are available:
- **Unbalanced** (default): a three-phase, four-wire model. Each phase keeps its own loads, cable impedance and substation voltage. Single-phase customers draw unequal currents on the three phases, and the return current flows through the neutral conductor, which shifts the neutral potential along the feeder and with it every customer's phase-to-neutral voltage — also on phases with little load.
- **Balanced**: every node's total power is spread evenly over the three phases, with a symmetric substation voltage. Useful as a simple reference.

The **computed** voltages are stored separately from the **measured** ones (meter readings). Results always show both side by side, so the model can be checked against the meters.

Use this agent to get voltage profiles under real load conditions, to detect voltage violations, to study the effect of phase unbalance, or to check how well the grid model matches the measurements.

**Prerequisites:** The grid must be registered (Grid Agent), with a node "PT", and measurements must be uploaded (Measurements Agent), including the PT voltage.

---

## Functions

### run_power_flow

Use this when you want to simulate the electrical state of the grid, compute voltages at all nodes, or check for voltage violations. This runs the simulation and returns a summary — retrieve the voltages with `get_power_flow_results`.

**Parameters:**
- `grid_id` (string, required): The grid to simulate
- `mode` (string, optional, default `"unbalanced"`): `"unbalanced"` or `"balanced"`
- `neutral_impedance_ratio` (number, optional, default `1.0`): Unbalanced mode only. Neutral conductor impedance relative to the phase conductors (1.0 for a cable with a full-size neutral such as 4x95; 0 for an ideal neutral, which makes the phases independent)
- `start_time` (string, optional): Start of the simulation window (ISO 8601, e.g. `"2025-07-01T00:00:00"`)
- `end_time` (string, optional): End of the simulation window (ISO 8601)

**Returns:** A summary: number of timestamps solved, how many did not converge, how many were skipped for lack of a PT voltage, and nodes not connected to PT.

**Notes:**
- Each run replaces the grid's previous results
- If no time range is given, all measurement timestamps are used
- A grid with a loop (not radial) can't be solved and returns an error

**Example:**
```json
{
  "grid_id": "grid001",
  "mode": "unbalanced",
  "neutral_impedance_ratio": 1.0,
  "start_time": "2025-07-01T00:00:00",
  "end_time": "2025-07-01T23:00:00"
}
```

---

### get_power_flow_results

Use this when you want to retrieve computed voltages after running a simulation, check if any node has an undervoltage or overvoltage violation, analyse the voltage profile across the grid, or compare the computed voltages with the measured ones.

**Parameters:**
- `grid_id` (string, required): The grid whose results to retrieve
- `phase` (string, optional): Only this phase — `"R"`, `"S"`, `"T"`, or `"N"` (neutral-to-ground voltage, unbalanced runs only). Default: all phases

**Returns:** The run settings, per-node errors of the computed voltages against the measured ones, and per-node, per-phase, per-timestamp results.

**How to interpret:**
- `computed`: the power flow's voltage (`voltage_magnitude` in V, `voltage_angle` in degrees). Phases R/S/T are phase-to-neutral voltages
- `measured`: the meter's voltage for the same node, phase and time, or null where there is no meter (and always for N)
- `errors`: per node and phase, the mean absolute error, maximum error and bias (computed − measured) of the voltage magnitude. Small errors mean the grid model (topology, cable impedances, phase assignment) matches reality; a large error at one node often points to a wrong phase assignment or cable data
- `converged: false` marks timestamps where the solver did not converge — treat those values as unreliable
- Nominal LV voltage is 230 V; **acceptable range** is ±10 % (207 V to 253 V)
- Below 207 V → undervoltage (heavy load, EV charging); above 253 V → overvoltage (PV generation exceeding local consumption)
- Node "PT" is the reference: its computed voltage equals its measured voltage by construction
- Phase N shows how far the neutral rises above ground; it grows with phase unbalance and distance from PT

**Example:**
```json
{
  "grid_id": "grid001",
  "phase": "R"
}
```
