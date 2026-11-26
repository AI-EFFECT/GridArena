STATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_training_state_data",
            "description": "Retrieve historical measurement data for training a state estimation algorithm. Returns masked grid and node identifiers with measurements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "noise_difficulty": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "Noise level added to measurements. Default is clean."
                    },
                    "observability": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "Controls how many nodes are observable. Default is medium."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_state_estimation_submission_template",
            "description": "Get the blank Python submission script for the state estimation benchmark: the predict function to fill in, with its input/output format documented.",
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
            "name": "submit_state_estimation_script",
            "description": "Submit a Python script defining predict(task) -> {series_key: {'voltage_magnitude', 'voltage_angle'}}: given the known series at a timestamp, the unknown series and their history, estimate the unknown voltages. The server runs it on the hidden estimation tasks and scores it (mean squared voltage error plus mean |sin| of the angle error, lower is better); returns a submission_id to check with get_state_estimation_submission.",
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
                    "noise_difficulty": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "Noise level of the test measurements. Default is clean."
                    },
                    "observability": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "How many series are unknown and how much of their history is hidden. Default is medium."
                    }
                },
                "required": [
                    "submission_name",
                    "code"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_state_estimation_submission",
            "description": "Get the status (queued, running, completed, failed), score and log of a state estimation script submission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_id": {
                        "type": "string",
                        "description": "The submission_id returned by submit_state_estimation_script"
                    }
                },
                "required": [
                    "submission_id"
                ]
            }
        }
    }
]
