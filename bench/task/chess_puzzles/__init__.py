from pathlib import Path

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import csv_dataset
from inspect_ai.scorer import exact
from inspect_ai.solver import prompt_template

from bench.solver.qa_agent import qa_agent

PUZZLES_FILENAME = str(Path(__file__).parent / "puzzles.csv")

QUESTION_TEMPLATE = (
    "Below is a FEN string representing the state of a chess game. Analyze the position and "
    "determine the best next move for the player whose turn it is.\n"
    "\n"
    "{prompt}"
)


SUBMIT_DESCRIPTION = (
    "Submit the best next move as a four-character string stating the starting and ending squares "
    "of the piece to be moved, e.g. b1c3. The conversation will end immediately."
)


FINAL_ANSWER_PROMPT = (
    "You have run out of output tokens. Do not reason any further: call `submit_answer` now with "
    "your best move."
)


@task(name="Chess Puzzles")
def chess_puzzles(
    *,
    epochs: int = 1,
    max_output_tokens: int,
    compaction_threshold_tokens: int,
) -> Task:
    """Single-move chess puzzles, answered through qa_agent.

    2.0.0 replaced the one-call harness (where a response cut off at the model's output token limit
    scored as no answer) with qa_agent, which continues cut-off responses and forces a final answer
    if the token budget runs out.
    """
    dataset = csv_dataset(PUZZLES_FILENAME, auto_id=True)

    plan = [
        prompt_template(QUESTION_TEMPLATE),
        qa_agent(
            max_output_tokens=max_output_tokens,
            compaction_threshold_tokens=compaction_threshold_tokens,
            submit_description=SUBMIT_DESCRIPTION,
            final_answer_prompt=FINAL_ANSWER_PROMPT,
        ),
    ]

    reducers = ["mean"]

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=exact(),
        epochs=Epochs(epochs, reducers),
        metadata={
            "epochai-benchmark-version": "2.0.0",
            "inspect-log-public": True,
        },
    )
