from typing import Any, Callable

from inspect_ai.model import Model
from inspect_ai.tool import ToolDef, ToolFunction


def answer_submission_tool(submit: Callable[..., Any]) -> ToolDef:
    return ToolDef(
        tool=submit,
        name="submit",
        description="Submit the answer",
        parameters={"answer": "The answer"},
    )


async def extract_model_answer(
    grader_model: Model,
    extraction_prompt: str,
    submission_tool: ToolDef,
    argument_name: str = "answer",
) -> Any:
    extraction_output = await grader_model.generate(
        extraction_prompt,
        tools=[submission_tool],
        tool_choice=ToolFunction(name="submit"),
    )

    if not extraction_output.message.tool_calls:
        raise RuntimeError("Grader model did not correctly extract an answer")

    tool_call = extraction_output.message.tool_calls[0]
    if argument_name not in tool_call.arguments:
        raise RuntimeError("Grader model did not correctly extract an answer")

    return tool_call.arguments[argument_name]
