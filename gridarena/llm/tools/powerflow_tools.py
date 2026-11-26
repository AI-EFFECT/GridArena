POWERFLOW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_power_flow",
            "description": "Run the power flow for a grid over its measurement timestamps (optionally a datetime range). All three phases are solved together; the results replace the grid's previous results and can be retrieved afterwards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_id": {
                        "type": "string",
                        "description": "The grid ID to run power flow on"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["unbalanced", "balanced"],
                        "description": "unbalanced (default): each phase keeps its own loads, cable impedance and substation voltage, with the neutral conductor modelled. balanced: each node's total power spread evenly over the phases."
                    },
                    "neutral_impedance_ratio": {
                        "type": "number",
                        "description": "Unbalanced mode only: neutral conductor impedance relative to the phase impedance. Default 1.0 (same as the phases); 0 = ideal neutral."
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Optional start datetime in ISO format e.g. 2025-07-01T00:00:00"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "Optional end datetime in ISO format e.g. 2025-07-01T23:59:59"
                    }
                },
                "required": ["grid_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_power_flow_results",
            "description": "Retrieve a grid's stored power flow results: the computed voltage magnitude and angle per node, phase and timestamp, next to the measured voltage where a meter measured it, plus the error of the computed voltages against the measured ones per node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grid_id": {
                        "type": "string",
                        "description": "The grid ID to retrieve results for"
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["R", "S", "T", "N"],
                        "description": "Optional: only this phase (N = neutral-to-ground voltage of unbalanced runs). Default: all phases."
                    }
                },
                "required": ["grid_id"]
            }
        }
    }
]
