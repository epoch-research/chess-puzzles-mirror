"""Agentic scaffold for non-agentic QA-style evals."""

import math

from inspect_ai.log._transcript import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    CompactionNative,
    CompactionSummary,
    Model,
    compaction,
    execute_tools,
    get_model,
)
from inspect_ai.model._compaction.types import CompactionStrategy
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, think, tool, tool_with

SUBMIT_NAME = "submit_answer"

PROMPT_TEMPLATE = (
    "Please answer the question below. Work through the problem carefully and then submit your "
    "final answer using the `submit_answer` tool. Use the `think` tool to record your reasoning. "
    "Feel free to use it as much as you want.\n"
    "\n"
    "You have a maximum of {max_output_tokens:,} output tokens. I will remind you of how many "
    "tokens you have remaining.\n"
    "\n"
    "**Here is the question:**\n"
    "\n"
    "{question}"
)


@tool
def submit_answer() -> Tool:
    async def execute(answer: str) -> str:
        """Submit your final answer. The conversation will end immediately.

        Args:
          answer: Your answer.
        """
        return answer

    return execute


def _extract_submission(tool_results: list[ChatMessage]) -> str | None:
    for result in tool_results:
        if isinstance(result, ChatMessageTool) and result.function == SUBMIT_NAME:
            return result.text
    return None


def _pick_compaction_strategy(model: Model, threshold: int) -> CompactionStrategy:
    """Native compaction for OpenAI; summary compaction for everyone else.

    Native compaction (`responses.compact`) is OpenAI-only: Anthropic's equivalent folds
    the user prompt into a model-generated compaction block instead of preserving it
    verbatim, and errored server-side when last tried (Inspect's token accounting
    disagreed with Anthropic's); xAI's OpenAI-compatible endpoint doesn't implement it.
    Matched by exact class name rather than isinstance because `EpochxAIAPI` subclasses
    `EpochOpenAIAPI` and must fall through to summary compaction.
    """
    if type(model.api).__name__ in ("OpenAIAPI", "EpochOpenAIAPI"):
        return CompactionNative(threshold=threshold)
    return CompactionSummary(threshold=threshold)


@solver
def qa_agent(
    *,
    max_output_tokens: int,
    compaction_threshold_tokens: int,
    submit_description: str | None = None,
) -> Solver:
    """Agentic scaffold for single-question evals.

    The sample's question (`state.user_prompt.text`) is wrapped in `PROMPT_TEMPLATE`. The model has
    the `think` and `submit_answer` tools and loops until it submits or reaches a budget backstop.
    On submit, the submitted string is written to `state.output.completion` so the scorer can see
    the answer without explicit plumbing.

    Args:
      max_output_tokens: Output token budget across all model turns in the loop. When exhausted, the
        loop exits and the last assistant message is used as the final output.
      compaction_threshold_tokens: Input token threshold that triggers compaction of the message
        history.
      submit_description: Optional override for the submit tool's description.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.user_prompt.text = PROMPT_TEMPLATE.format(
            max_output_tokens=max_output_tokens,
            question=state.user_prompt.text,
        )

        submit_tool = submit_answer()
        if submit_description is not None:
            submit_tool = tool_with(submit_tool, description=submit_description)
        state.tools = [think(), submit_tool]

        model = get_model()
        compact = compaction(
            strategy=_pick_compaction_strategy(model, compaction_threshold_tokens),
            prefix=state.messages[:],
            tools=state.tools,
            model=model,
        )

        # Doom loop protection: the model should be spending many tokens per turn if it's actually
        # thinking. Cut it off at budget/2000 turns because it's probably stuck repeatedly
        # outputting a small number of tokens per turn (and racking up input tokens).
        max_turns = math.ceil(max_output_tokens / 2_000)

        output_tokens_used = 0
        turns = 0

        while not state.completed:
            if turns >= max_turns:
                transcript().info(f"qa_agent: terminated — reached turn cap of {max_turns}")
                break
            turns += 1

            input_messages, supplemental = await compact.compact_input(state.messages)
            if supplemental is not None:
                state.messages.append(supplemental)
                # Add token budget reminder that may have been summarized away. Also nudge it to
                # submit so it doesn't get into a loop of redoing its analysis.
                reminder = ChatMessageUser(
                    content=(
                        f"You have used {output_tokens_used:,}/{max_output_tokens:,} output "
                        f"tokens. If your analysis is complete, call the `submit_answer` tool."
                    )
                )
                state.messages.append(reminder)
                input_messages.append(reminder)

            state.output = await model.generate(input=input_messages, tools=state.tools)
            await compact.record_output(input_messages, state.output)
            state.messages.append(state.output.message)

            turn_output_tokens = (
                state.output.usage.output_tokens if state.output.usage is not None else 0
            )
            output_tokens_used += turn_output_tokens

            if state.output.stop_reason == "model_length":
                transcript().info(
                    "qa_agent: terminated — model context window was exceeded "
                    '("stop_reason"="model_length") before the submit tool was called'
                )
                break

            if state.output.stop_reason == "content_filter":
                transcript().info(
                    "qa_agent: terminated — model response was blocked by a content filter "
                    "before the submit tool was called"
                )
                break

            if state.output.message.tool_calls:
                tool_results, _ = await execute_tools([state.output.message], state.tools)
                state.messages.extend(tool_results)

                answer = _extract_submission(tool_results)
                if answer is not None:
                    state.output.completion = answer
                    break

            remaining = max_output_tokens - output_tokens_used
            if remaining <= 0:
                transcript().info(
                    f"qa_agent: output token budget of {max_output_tokens:,} exhausted"
                )
                break

            if turn_output_tokens <= 0:
                transcript().info(
                    "qa_agent: terminated — model made no progress against the output token budget"
                )
                break

            # The next generate() requires the conversation to end with a user message: some
            # providers (e.g. Anthropic adaptive-thinking models) reject a trailing assistant
            # message as unsupported prefill.
            message = f"You have used {output_tokens_used:,}/{max_output_tokens:,} output tokens."
            if not state.output.message.tool_calls:
                message += (
                    " Your previous response ended without a tool call (it may have been cut"
                    " off). Please continue, and call `submit_answer` when you have your answer."
                )
            state.messages.append(ChatMessageUser(content=message))

        return state

    return solve
