from typing import Any, Literal

from inspect_ai.model import Model
from inspect_ai.tool import ToolDef, ToolFunction, ToolParam, ToolParams


SubmissionParamType = Literal["string", "integer", "number", "boolean", "array", "object"]


def nullable_tool_param(
    param_type: SubmissionParamType,
    description: str = "The answer",
) -> ToolParam:
    return ToolParam(
        description=description,
        anyOf=[
            ToolParam(type=param_type),
            ToolParam(type="null"),
        ],
    )


def answer_submission_tool(
    answer_param: ToolParam,
    argument_name: str = "answer",
    description: str = "Submit the answer",
) -> ToolDef:
    return ToolDef(
        tool=lambda _: None,
        name="submit",
        description=description,
        parameters=ToolParams(
            properties={argument_name: answer_param},
            required=[argument_name],
        ),
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
