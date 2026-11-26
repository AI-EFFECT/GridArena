VOLTAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_training_voltage_control_data",
            "description": "Retrieve training snapshots with reference solutions for voltage control. Returns measurements and optimal solutions for each grid and timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "voltage_limit": {
                        "type": "number",
                        "description": "Voltage threshold used to select snapshots with violations e.g. 230.0"
                    },
                    "scenario": {
                        "type": "string",
                        "enum": [
                            "Overvoltages",
                            "Undervoltages"
                        ],
                        "description": "Whether to retrieve overvoltage or undervoltage scenarios"
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "Filters snapshots by hardness. Default is clean."
                    }
                },
                "required": [
                    "voltage_limit",
                    "scenario"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_voltage_control_submission_template",
            "description": "Get the blank Python submission script for the voltage control benchmark: the predict function to fill in, with its input/output format documented.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_voltage_control_script",
            "description": "Submit a Python script defining predict(snapshot) -> [{'node_id', 'phase', 'corrected_voltage', 'adjusted_power_active', 'adjusted_power_reactive'}, ...]: given a snapshot with voltage violations, return corrective actions in the format of the training reference solutions. The server runs it on the hidden test snapshots and scores it (a 0-1 score (higher is better) rewarding matching the reference voltages with the least change in active power); returns a submission_id to check with get_voltage_control_submission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_name": {
                        "type": "string",
                        "description": "Label for this attempt, e.g. 'run_001' (letters, digits, '_', '-', '.')"
                    },
                    "code": {
                        "type": "string",
                        "description": "Full source of the Python script, based on the submission template"
                    },
                    "voltage_limit": {
                        "type": "number",
                        "description": "Voltage threshold (V) that selects the snapshots with violations, e.g. 241.5"
                    },
                    "scenario": {
                        "type": "string",
                        "enum": [
                            "Overvoltages",
                            "Undervoltages"
                        ],
                        "description": "Type of violation. Default is Overvoltages."
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "Filters the snapshots by hardness. Default is clean."
                    }
                },
                "required": [
                    "submission_name",
                    "code",
                    "voltage_limit"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_voltage_control_submission",
            "description": "Get the status (queued, running, completed, failed), score and log of a voltage control script submission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_id": {
                        "type": "string",
                        "description": "The submission_id returned by submit_voltage_control_script"
                    }
                },
                "required": [
                    "submission_id"
                ]
            }
        }
    }
]
