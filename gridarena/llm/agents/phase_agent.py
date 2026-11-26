from gridarena.llm.base_agent import BaseAgent
from gridarena.routers.phase_identification_benchmark import (
    get_phase_submission,
    get_submission_template_source,
    get_training_phase_identification_data,
    submit_phase_script,
)


class PhaseAgent(BaseAgent):

    async def _training(self, difficulty: str = "clean"):
        self._record("get_training_phase_data", difficulty=difficulty)
        return await self._call(get_training_phase_identification_data, difficulty=difficulty)

    async def _template(self):
        self._record("get_phase_submission_template")
        return await self._call(get_submission_template_source)

    async def _submit(self, submission_name: str, code: str, difficulty: str = "clean"):
        self._record("submit_phase_script", submission_name=submission_name, difficulty=difficulty)
        return await self._call(
            submit_phase_script, source=code.encode("utf-8"), filename="submission.py",
            submission_name=submission_name, difficulty=difficulty,
        )

    async def _status(self, submission_id: str):
        self._record("get_phase_submission", submission_id=submission_id)
        return await self._call(get_phase_submission, submission_id=submission_id)

    async def handle(self, tool_name: str, tool_args: dict):
        if tool_name == "get_training_phase_data":
            return await self._training(difficulty=tool_args.get("difficulty", "clean"))
        elif tool_name == "get_phase_submission_template":
            return await self._template()
        elif tool_name == "submit_phase_script":
            return await self._submit(
                submission_name=tool_args["submission_name"], code=tool_args["code"],
                difficulty=tool_args.get("difficulty", "clean"),
            )
        elif tool_name == "get_phase_submission":
            return await self._status(submission_id=tool_args["submission_id"])
        return await super().handle(tool_name, tool_args)
