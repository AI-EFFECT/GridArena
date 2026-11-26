PHASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_training_phase_data",
            "description": "Retrieve historical measurement data for training a phase detection algorithm. Returns labelled data with true phases visible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "string",
                        "enum": ["clean", "easy", "medium", "hard"],
                        "description": "Difficulty level of the data corruption profile. Default is clean."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_phase_submission_template",
            "description": "Get the blank Python submission script for the phase identification benchmark: the predict(measurements) function to fill in, with its input/output format documented.",
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
            "name": "submit_phase_script",
            "description": "Submit a Python script defining predict(measurements) -> {series_key: 'R'|'S'|'T'}. The server runs it on the hidden test grids and scores it; returns a submission_id to check with get_phase_submission.",
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
                        "enum": ["clean", "easy", "medium", "hard"],
                        "description": "Noise level of the test data the script is run on. Default is clean."
                    }
                },
                "required": ["submission_name", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_phase_submission",
            "description": "Get the status (queued, running, completed, failed), accuracy score and log of a phase identification script submission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_id": {
                        "type": "string",
                        "description": "The submission_id returned by submit_phase_script"
                    }
                },
                "required": ["submission_id"]
            }
        }
    }
]
