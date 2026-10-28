"""The `submit_answer` tool shared by qa_agent and structured_answer."""

from inspect_ai.model import ChatMessage, ChatMessageTool, execute_tools
from inspect_ai.solver import TaskState
from inspect_ai.tool import Tool, tool

SUBMIT_NAME = "submit_answer"


@tool
def submit_answer() -> Tool:
    async def execute(answer: str) -> str:
        """Submit your final answer. The conversation will end immediately.

        Args:
          answer: Your answer.
        """
        return answer

    return execute


def _extract_submission(tool_results: list[ChatMessage], ignore_rejected: bool) -> str | None:
    for result in tool_results:
        if isinstance(result, ChatMessageTool) and result.function == SUBMIT_NAME:
            if ignore_rejected and result.error is not None:
                continue
            return result.text
    return None


async def submit_if_called(state: TaskState, ignore_rejected: bool = False) -> bool:
    """Runs the model's tool calls; on a submission, sets it as the completion and returns True.

    With `ignore_rejected`, a `submit_answer` call that the tool rejects isn't a submission. Without
    it, as in qa_agent, the first `submit_answer` call is the submission even if rejected.
    """
    if not state.output.message.tool_calls:
        return False
    tool_results, _ = await execute_tools([state.output.message], state.tools)
    state.messages.extend(tool_results)
    answer = _extract_submission(tool_results, ignore_rejected)
    if answer is None:
        return False
    state.output.completion = answer
    return True
