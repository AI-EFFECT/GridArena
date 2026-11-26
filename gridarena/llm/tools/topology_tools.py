TOPOLOGY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_training_topology_data",
            "description": "Retrieve historical measurement and grid topology data for training a topology discovery algorithm. True topology is visible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "Difficulty level of data corruption. Default is clean."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_topology_submission_template",
            "description": "Get the blank Python submission script for the topology discovery benchmark: the predict function to fill in, with its input/output format documented.",
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
            "name": "submit_topology_script",
            "description": "Submit a Python script defining predict(grid) -> [[node_key, node_key], ...]: given every node key of a grid, its substation and the measurements per node, return the grid's edges. The server runs it on the hidden test grids and scores it (F1 of the predicted edges); returns a submission_id to check with get_topology_submission.",
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
                    "difficulty": {
                        "type": "string",
                        "enum": [
                            "clean",
                            "easy",
                            "medium",
                            "hard"
                        ],
                        "description": "Noise level of the test measurements. Default is clean."
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
            "name": "get_topology_submission",
            "description": "Get the status (queued, running, completed, failed), score and log of a topology discovery script submission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_id": {
                        "type": "string",
                        "description": "The submission_id returned by submit_topology_script"
                    }
                },
                "required": [
                    "submission_id"
                ]
            }
        }
    }
]
