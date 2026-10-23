"""Solvers that take a model's answer through the `submit_answer` tool."""

import random
from collections import Counter
from typing import Callable, Mapping

from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolParam, ToolParams

from bench.solver.forced_answer import forced_tool_choice
from bench.solver.submission import SUBMIT_NAME, submit_answer, submit_if_called

DEFAULT_DESCRIPTION = "Call this tool to submit your answer."
SUBMIT_REMINDER = "Please submit your answer by calling the `submit_answer` tool."
SUBMISSION_KEY = "structured_answer:submission"
MAX_SUBMIT_ATTEMPTS = 3


@solver
def structured_answer(
    field: Callable[[TaskState], ToolParam],
    description: str = DEFAULT_DESCRIPTION,
) -> Solver:
    """Takes the model's answer through a `submit_answer` tool whose argument is `field(state)`.

    If the tool rejects every call in a response, the model sees the errors and tries again, up to
    MAX_SUBMIT_ATTEMPTS times. A response without a call (`tool_choice` can't always force one) is
    followed by SUBMIT_REMINDER and also counts as an attempt; a refusal ends the attempts.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.tools = [
            ToolDef(
                submit_answer(),
                name=SUBMIT_NAME,
                description=description,
                parameters=ToolParams(properties={"answer": field(state)}, required=["answer"]),
            ).as_tool()
        ]
        state.tool_choice = forced_tool_choice(str(state.model), SUBMIT_NAME)
        submitted = False
        for attempt in range(MAX_SUBMIT_ATTEMPTS):
            if attempt > 0 and not state.output.message.tool_calls:
                state.messages.append(ChatMessageUser(content=SUBMIT_REMINDER))
            state = await generate(state, tool_calls="none")
            submitted = await submit_if_called(state, ignore_rejected=True)
            if submitted or state.output.stop_reason == "content_filter":
                break
        set_submitted_answer(state, state.output.completion if submitted else "")
        return state

    return solve


def submitted_answer(state: TaskState) -> str:
    """Returns the answer submitted through `structured_answer`, or "" if there was none.

    Scorers read it from the store because a log keeps the store as written, but refills an empty
    `state.output.completion` from the model's text when it is read back.
    """
    return state.store.get(SUBMISSION_KEY, "")


def set_submitted_answer(state: TaskState, answer: str) -> None:
    """Records `answer` as the submission and as the output completion."""
    state.output.completion = answer
    state.store.set(SUBMISSION_KEY, answer)


def choice_field(question: str, options: Mapping[str, str | None]) -> ToolParam:
    """Returns a string argument that must be one of `options`.

    Args:
      question: The argument's description.
      options: Each option, mapped to its description or to None.
    """
    if not any(options.values()):
        return ToolParam(type="string", description=question, enum=list(options))
    return ToolParam(
        type="string",
        description=question,
        anyOf=[
            ToolParam(type="string", enum=[value], description=description)
            for value, description in options.items()
        ],
    )


def sample_random(state: TaskState) -> random.Random:
    """Returns a random generator seeded by the sample ID and epoch."""
    return random.Random(f"{state.sample_id}-{state.epoch}")


@solver
def single_choice_answer(
    options: Callable[[TaskState], list[str]],
    *,
    shuffle: bool = False,
    description: str = DEFAULT_DESCRIPTION,
) -> Solver:
    """Takes the model's answer as one of `options(state)`.

    Args:
      options: Returns the sample's options, in the order to offer them.
      shuffle: Shuffle the options per sample and epoch.
      description: The `submit_answer` tool's description.
    """

    def field(state: TaskState) -> ToolParam:
        values = list(options(state))
        repeated = sorted(value for value, count in Counter(values).items() if count > 1)
        if repeated:
            raise ValueError(f"Sample {state.sample_id} offers {repeated} more than once.")
        if shuffle:
            sample_random(state).shuffle(values)
        return choice_field("Your answer.", dict.fromkeys(values))

    return structured_answer(field, description)


def requires_submission(inner: Scorer) -> Scorer:
    """Scores with `inner`, or NOANSWER if nothing was submitted through `structured_answer`."""

    async def score(state: TaskState, target: Target) -> Score | None:
        if not submitted_answer(state):
            return Score(value=NOANSWER, answer="", explanation="No submission was provided.")
        return await inner(state, target)

    return score


@scorer(metrics=[accuracy(), stderr()])
def exact_submission() -> Scorer:
    """Scores the submission by exact match with a target, or NOANSWER if nothing was submitted.

    Unlike Inspect's `exact()`, it doesn't normalize case, punctuation, articles or numbers.
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = submitted_answer(state)
        return Score(value=CORRECT if answer in target.target else INCORRECT, answer=answer)

    return requires_submission(score)
