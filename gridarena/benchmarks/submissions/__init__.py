"""Code-submission evaluation shared by the benchmarks: users upload a script
defining a fixed function, the server runs it on the hidden test data in a
sandbox and scores its predictions."""

from .packaging import SubmissionError
from .run_manager import BenchmarkTask, PreparedInputs, SubmissionRegistry, submission_registry
