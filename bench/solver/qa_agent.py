"""Agentic scaffold for non-agentic QA-style evals."""

import math
from typing import TypeGuard

from inspect_ai.log._transcript import transcript
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    CompactionNative,
    CompactionSummary,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Model,
    ModelOutput,
    compaction,
    get_model,
)
from inspect_ai.model._compaction.types import CompactionStrategy
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import tool_with

from bench.solver.forced_answer import forced_tool_choice, lowest_reasoning_effort
from bench.solver.submission import SUBMIT_NAME, submit_answer, submit_if_called

# Output cap for the forced final answer. The answer itself is a few tokens, but at their lowest
# reasoning effort some models (those without `none`) still reason briefly first.
FINAL_ANSWER_MAX_TOKENS = 16_000

# Every call asks for the model's full reasoning history, so that providers which drop replayed
# reasoning by default (Zhipu, Alibaba) keep it: the model can build on its earlier turns.
FULL_REASONING_HISTORY = GenerateConfig(reasoning_history="all")

# OpenAI rejects a replayed `encrypted_content` longer than this (400 `string_above_max_length`).
# A response cut off at 128k tokens of reasoning can encrypt to ~1.5M characters; such reasoning is
# replaced in the history by OpenAI's summary of it, as visible text, so the model can continue.
MAX_REPLAYED_ENCRYPTED_REASONING_CHARS = 1_048_576
REASONING_SUMMARY_HEADER = (
    "Summary of my reasoning so far (the full reasoning was too long to keep):"
)

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


def _is_openai(model: Model) -> bool:
    """Matched by exact class name rather than isinstance because `EpochxAIAPI` subclasses
    `EpochOpenAIAPI` and must not count as OpenAI."""
    return type(model.api).__name__ in ("OpenAIAPI", "EpochOpenAIAPI")


def _caps_replayed_encrypted_reasoning(model: Model) -> bool:
    return _is_openai(model)


def _is_oversized_reasoning(content: object) -> TypeGuard[ContentReasoning]:
    return (
        isinstance(content, ContentReasoning)
        and content.redacted
        and len(content.reasoning) > MAX_REPLAYED_ENCRYPTED_REASONING_CHARS
    )


def _summarize_oversized_reasoning(message: ChatMessageAssistant) -> int:
    """Replace oversized encrypted reasoning with its summary as text; return how many."""
    if not isinstance(message.content, list):
        return 0
    oversized = [c for c in message.content if _is_oversized_reasoning(c)]
    if any(not c.summary for c in oversized):
        raise ValueError(
            "Encrypted reasoning is too long to replay and has no summary to replace it with."
        )
    message.content = [
        (
            ContentText(text=f"{REASONING_SUMMARY_HEADER}\n\n{c.summary}")
            if _is_oversized_reasoning(c)
            else c
        )
        for c in message.content
    ]
    return len(oversized)


def _drop_cut_off_tool_calls(output: ModelOutput) -> None:
    """Drop the tool calls of a response that the output token cap cut off.

    Generation stopped mid-response, so a tool call in it may be truncated: OpenAI returns the
    partial arguments (e.g. `{"answer":"b2b3`), Anthropic's SDK parses partial JSON into whatever
    fields were complete. Executing it would take a broken or cut-short `submit_answer` as the
    submission. Without the calls the response reads as cut off before calling a tool, so the
    model is asked to continue (and the calls are not replayed in its history).
    """
    if output.stop_reason != "max_tokens" or not output.message.tool_calls:
        return
    transcript().info(
        f"qa_agent: dropped {len(output.message.tool_calls)} tool call(s) from a response cut off "
        '("stop_reason"="max_tokens") before it finished'
    )
    output.message.tool_calls = None


def _pick_compaction_strategy(model: Model, threshold: int) -> CompactionStrategy:
    """Native compaction for OpenAI; summary compaction for everyone else.

    Native compaction (`responses.compact`) is OpenAI-only: Anthropic's equivalent folds
    the user prompt into a model-generated compaction block instead of preserving it
    verbatim, and errored server-side when last tried (Inspect's token accounting
    disagreed with Anthropic's); xAI's OpenAI-compatible endpoint doesn't implement it.
    """
    if _is_openai(model):
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

    On OpenAI, encrypted reasoning too long to send back (over
    `MAX_REPLAYED_ENCRYPTED_REASONING_CHARS`) is replaced in the history by its summary, as text;
    `metadata["reasoning_replaced_by_summary"]` counts those replacements.

    Tool calls in a response cut off by the output token cap (`stop_reason="max_tokens"`) are
    dropped unexecuted, since they may be truncated; the model is asked to continue instead.

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
        reasoning_summaries = 0

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
            if _caps_replayed_encrypted_reasoning(model):
                summarized = _summarize_oversized_reasoning(state.output.message)
                if summarized:
                    transcript().info(
                        "qa_agent: encrypted reasoning too long to replay (over "
                        f"{MAX_REPLAYED_ENCRYPTED_REASONING_CHARS:,} characters); replaced by its "
                        "summary"
                    )
                    reasoning_summaries += summarized

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

            _drop_cut_off_tool_calls(state.output)
            if await submit_if_called(state):
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
        state.metadata["reasoning_replaced_by_summary"] = reasoning_summaries
        if exhausted:
            model_name = str(state.model)
            state.messages.append(ChatMessageUser(content=final_answer_prompt))
            input_messages, _ = await compact.compact_input(state.messages)
            state.output = await model.generate(
                input=input_messages,
                tools=state.tools,
                tool_choice=forced_tool_choice(model_name, SUBMIT_NAME),
                config=FULL_REASONING_HISTORY.merge(
                    GenerateConfig(
                        max_tokens=FINAL_ANSWER_MAX_TOKENS,
                        reasoning_effort=lowest_reasoning_effort(model_name),
                    )
                ),
            )
            await compact.record_output(input_messages, state.output)
            state.messages.append(state.output.message)
            _drop_cut_off_tool_calls(state.output)
            await submit_if_called(state)

        return state

    return solve
