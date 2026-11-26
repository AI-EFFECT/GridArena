from gridarena.llm.base_agent import BaseAgent
from gridarena.routers.voltage_control_benchmark import (
    get_submission_template_source,
    get_training_voltage_control_data,
    get_voltage_control_submission,
    submit_voltage_control_script,
)


class VoltageAgent(BaseAgent):

    async def _training(self, voltage_limit: float, scenario: str = "Overvoltages", difficulty: str = "clean"):
        self._record("get_training_voltage_control_data")
        return await self._call(
            get_training_voltage_control_data,
            voltage_limit=voltage_limit, Scenario=scenario, difficulty=difficulty,
        )

    async def _template(self):
        self._record("get_voltage_control_submission_template")
        return await self._call(get_submission_template_source)

    async def _submit(self, submission_name: str, code: str, voltage_limit: float,
                      scenario: str = "Overvoltages", difficulty: str = "clean"):
        self._record("submit_voltage_control_script", submission_name=submission_name)
        return await self._call(
            submit_voltage_control_script, source=code.encode("utf-8"), filename="submission.py",
            submission_name=submission_name, voltage_limit=voltage_limit, scenario=scenario,
            difficulty=difficulty,
        )

    async def _status(self, submission_id: str):
        self._record("get_voltage_control_submission", submission_id=submission_id)
        return await self._call(get_voltage_control_submission, submission_id=submission_id)

    async def handle(self, tool_name: str, tool_args: dict):
        if tool_name == "get_training_voltage_control_data":
            return await self._training(
                voltage_limit=tool_args["voltage_limit"],
                scenario=tool_args.get("scenario", "Overvoltages"),
                difficulty=tool_args.get("difficulty", "clean"),
            )
        elif tool_name == "get_voltage_control_submission_template":
            return await self._template()
        elif tool_name == "submit_voltage_control_script":
            return await self._submit(
                submission_name=tool_args["submission_name"], code=tool_args["code"],
                voltage_limit=tool_args["voltage_limit"],
                scenario=tool_args.get("scenario", "Overvoltages"),
                difficulty=tool_args.get("difficulty", "clean"),
            )
        elif tool_name == "get_voltage_control_submission":
            return await self._status(submission_id=tool_args["submission_id"])
        return await super().handle(tool_name, tool_args)
