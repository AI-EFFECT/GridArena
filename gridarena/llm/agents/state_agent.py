from gridarena.llm.base_agent import BaseAgent
from gridarena.routers.state_estimation_benchmark import (
    get_state_estimation_submission,
    get_submission_template_source,
    get_training_state_data,
    submit_state_estimation_script,
)


class StateAgent(BaseAgent):

    async def _training(self, noise_difficulty: str = "clean", observability: str = "medium"):
        self._record("get_training_state_data")
        return await self._call(
            get_training_state_data,
            noise_difficulty=noise_difficulty, observability=observability,
        )

    async def _template(self):
        self._record("get_state_estimation_submission_template")
        return await self._call(get_submission_template_source)

    async def _submit(self, submission_name: str, code: str,
                      noise_difficulty: str = "clean", observability: str = "medium"):
        self._record("submit_state_estimation_script", submission_name=submission_name)
        return await self._call(
            submit_state_estimation_script, source=code.encode("utf-8"), filename="submission.py",
            submission_name=submission_name, noise_difficulty=noise_difficulty, observability=observability,
        )

    async def _status(self, submission_id: str):
        self._record("get_state_estimation_submission", submission_id=submission_id)
        return await self._call(get_state_estimation_submission, submission_id=submission_id)

    async def handle(self, tool_name: str, tool_args: dict):
        if tool_name == "get_training_state_data":
            return await self._training(
                noise_difficulty=tool_args.get("noise_difficulty", "clean"),
                observability=tool_args.get("observability", "medium"),
            )
        elif tool_name == "get_state_estimation_submission_template":
            return await self._template()
        elif tool_name == "submit_state_estimation_script":
            return await self._submit(
                submission_name=tool_args["submission_name"], code=tool_args["code"],
                noise_difficulty=tool_args.get("noise_difficulty", "clean"),
                observability=tool_args.get("observability", "medium"),
            )
        elif tool_name == "get_state_estimation_submission":
            return await self._status(submission_id=tool_args["submission_id"])
        return await super().handle(tool_name, tool_args)
