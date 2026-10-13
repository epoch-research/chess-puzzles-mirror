"""Agentic scaffold for non-agentic QA-style evals."""

import math

from inspect_ai.log._transcript import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageTool,
    ChatMessageUser,
    CompactionNative,
    CompactionSummary,
    GenerateConfig,
    Model,
    compaction,
    execute_tools,
    get_model,
)
from inspect_ai.model._compaction.types import CompactionStrategy
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolFunction, tool, tool_with

from bench.solver.forced_answer import lowest_reasoning_effort, supports_forced_tool_choice

SUBMIT_NAME = "submit_answer"

# Output cap for the forced final answer. The answer itself is a few tokens, but at their lowest
# reasoning effort some models (those without `none`) still reason briefly first.
FINAL_ANSWER_MAX_TOKENS = 16_000

# Every call asks for the model's full reasoning history, so that providers which drop replayed
# reasoning by default (Zhipu, Alibaba) keep it: the model can build on its earlier turns.
FULL_REASONING_HISTORY = GenerateConfig(reasoning_history="all")

PROMPT_TEMPLATE = (
    "Please answer the question below. Work through the problem carefully and then submit your "
    "final answer using the `submit_answer` tool.\n"
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


async def _submit_if_called(state: TaskState) -> bool:
    """Execute the last message's tool calls; on a submission, make it the output completion."""
    if not state.output.message.tool_calls:
        return False
    tool_results, _ = await execute_tools([state.output.message], state.tools)
    state.messages.extend(tool_results)
    answer = _extract_submission(tool_results)
    if answer is None:
        return False
    state.output.completion = answer
    return True


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
    final_answer_prompt: str,
    submit_description: str | None = None,
) -> Solver:
    """Agentic scaffold for single-question evals.

    The sample's question (`state.user_prompt.text`) is wrapped in `PROMPT_TEMPLATE`. The model has
    the `submit_answer` tool and loops until it submits or reaches a budget backstop.
    On submit, the submitted string is written to `state.output.completion` so the scorer can see
    the answer without explicit plumbing.

    If the output token budget (or the turn cap) runs out before a submission, the model gets
    `final_answer_prompt` and one last call at its lowest reasoning effort, with `submit_answer`
    forced via `tool_choice` where the model supports it. `metadata["forced_final_answer"]` records
    whether that happened.

    Args:
      max_output_tokens: Output token budget across all model turns in the loop. When exhausted, the
        loop exits and the last assistant message is used as the final output.
      compaction_threshold_tokens: Input token threshold that triggers compaction of the message
        history.
      final_answer_prompt: User message asking for an immediate answer once the budget runs out.
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
        state.tools = [submit_tool]

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
        exhausted = False

        while not state.completed:
            if turns >= max_turns:
                transcript().info(f"qa_agent: terminated — reached turn cap of {max_turns}")
                exhausted = True
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

            state.output = await model.generate(
                input=input_messages, tools=state.tools, config=FULL_REASONING_HISTORY
            )
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

            if await _submit_if_called(state):
                break

            remaining = max_output_tokens - output_tokens_used
            if remaining <= 0:
                transcript().info(
                    f"qa_agent: output token budget of {max_output_tokens:,} exhausted"
                )
                exhausted = True
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

        state.metadata["forced_final_answer"] = exhausted
        if exhausted:
            model_name = str(state.model)
            state.messages.append(ChatMessageUser(content=final_answer_prompt))
            input_messages, _ = await compact.compact_input(state.messages)
            state.output = await model.generate(
                input=input_messages,
                tools=state.tools,
                tool_choice=(
                    ToolFunction(SUBMIT_NAME) if supports_forced_tool_choice(model_name) else "auto"
                ),
                config=FULL_REASONING_HISTORY.merge(
                    GenerateConfig(
                        max_tokens=FINAL_ANSWER_MAX_TOKENS,
                        reasoning_effort=lowest_reasoning_effort(model_name),
                    )
                ),
            )
            await compact.record_output(input_messages, state.output)
            state.messages.append(state.output.message)
            await _submit_if_called(state)

        return state

    return solve
