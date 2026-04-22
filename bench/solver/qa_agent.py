"""Simple agentic scaffold for non-agentic QA-style evals.

Wraps the sample's question in a scaffold prompt, gives the model Inspect's
built-in `think` tool plus a `submit_answer` tool (string answer), and loops
until either the model submits or an output-token budget is exhausted. Before
each model turn a user message reports how many output tokens remain.
"""

from inspect_ai.log._transcript import transcript
from inspect_ai.model import (
    ChatMessageTool,
    ChatMessageUser,
    execute_tools,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, think, tool, tool_with

SUBMIT_NAME = "submit_answer"

STUB_QUESTION_PROMPT = """You are answering the question below. Work through the problem carefully \
and submit a final answer.

You have two tools available:
- `think`: record a step of reasoning. Use it freely to work out your solution.
- `submit_answer`: submit your final answer as a string. Call this exactly once when you are ready; \
the conversation will end immediately.

You have a limited budget of output tokens; a reminder of how many remain will be sent before each \
of your turns.

Question:

{question}
"""


@tool
def submit_answer() -> Tool:
    async def execute(answer: str) -> str:
        """Submit your final answer. The conversation will end immediately.

        Args:
          answer: Your answer.
        """
        return answer

    return execute


def _extract_submission(tool_results: list[ChatMessageTool]) -> str | None:
    for result in tool_results:
        if isinstance(result, ChatMessageTool) and result.function == SUBMIT_NAME:
            return result.text
    return None


@solver
def qa_agent(
    *,
    max_output_tokens: int,
    submit_description: str | None = None,
) -> Solver:
    """Simple agentic scaffold for single-question evals.

    The sample's question (`state.user_prompt.text`) is wrapped in
    `STUB_QUESTION_PROMPT`. The model is given the `think` and `submit_answer`
    tools and looped until it submits or the output-token budget is exhausted.
    On a successful submit, the submitted string is written to
    `state.output.completion` so downstream scorers see the raw answer.

    Args:
      max_output_tokens: Output-token budget across all model turns in the
        loop. When exhausted, the loop exits and the last assistant message is
        used as the final output.
      submit_description: Optional override for the submit tool's description.
        The tool name is always `submit_answer`.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = STUB_QUESTION_PROMPT.format(
            question=state.user_prompt.text
        )

        submit_tool = submit_answer()
        if submit_description is not None:
            submit_tool = tool_with(submit_tool, description=submit_description)
        state.tools = [think(), submit_tool]

        output_tokens_used = 0

        while not state.completed:
            remaining = max_output_tokens - output_tokens_used
            if remaining <= 0:
                transcript().info(
                    f"qa_agent: output-token budget of {max_output_tokens} exhausted"
                )
                break

            state.messages.append(
                ChatMessageUser(
                    content=f"You have {remaining} output tokens remaining in your budget."
                )
            )

            state.output = await get_model().generate(
                input=state.messages, tools=state.tools
            )
            state.messages.append(state.output.message)

            if state.output.usage is not None:
                output_tokens_used += state.output.usage.output_tokens

            if state.output.stop_reason == "model_length":
                transcript().info(
                    "qa_agent: terminated — model context window exceeded "
                    "(stop_reason=model_length) before the submit tool was called"
                )
                break

            if not state.output.message.tool_calls:
                continue

            tool_results, _ = await execute_tools(
                [state.output.message], state.tools
            )
            state.messages.extend(tool_results)

            answer = _extract_submission(tool_results)
            if answer is not None:
                state.output.completion = answer
                break

        return state

    return solve
