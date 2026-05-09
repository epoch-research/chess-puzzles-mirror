from typing import Dict, Any

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    CORRECT,
    INCORRECT,
    NOANSWER,
    accuracy,
    exact,
    stderr,
)
from inspect_ai.solver import generate, prompt_template, TaskState

from bench.model import default_grader_model
from bench.solver.qa_agent import qa_agent
from bench.task.model_extracted_answer import (
    answer_submission_tool,
    extract_model_answer,
    nullable_tool_param,
)

SOLUTION_PROMPT_TEMPLATE = """Below is a FEN string representing the state of a chess game. Please analyze the position and determine the best next move for the player whose turn it is. You may think as much as you would like, but please conclude your response with the move expressed as a four-character string stating the starting and ending square of the piece to be moved, using the format "MOVE: b1c3" (without quotes).

{prompt}

"""

EXTRACTION_PROMPT_TEMPLATE = """**Task: Extract a student's answer to a single-move chess puzzle and submit the move that is their final answer.**

Instructions:
- Read the student's entire solution carefully to identify their final answer.
- The student is supposed to have concluded their solution with "MOVE: [answer]" where [answer] is given as a four-character string stating the starting and ending square of the piece to be moved, e.g. b1c3.
- If the solution appears in that format, extract the move and submit it as the final answer.
- Otherwise, do your best to identify the final answer.
- Use the tool provided to submit the extracted answer.
- If there are multiple moves in the solution, extract the one that represents the main answer to the puzzle.
- If no clear final answer can be identified, submit `None` instead.

Solution to extract from:

<student_solution>

{student_solution}

</student_solution>"""


@scorer(metrics=[accuracy(), stderr()])
def model_extracted_exact_match():
    async def score(state: TaskState, target: Target) -> Score:
        student_solution = state.output.completion

        extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(student_solution=student_solution)

        extracted_answer = await extract_model_answer(
            grader_model=default_grader_model(),
            extraction_prompt=extraction_prompt,
            submission_tool=answer_submission_tool(nullable_tool_param("string")),
        )

        if extracted_answer is None:
            return Score(
                value=NOANSWER,
                explanation="No clear answer was given",
            )

        is_correct = extracted_answer == target.text
        explanation = (
            f"The model answered {extracted_answer}, which is correct."
            if is_correct
            else f"The model answered {extracted_answer}, but the correct answer is {target.text}."
        )

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=str(extracted_answer),
            explanation=explanation,
        )

    return score


@task(name="Chess Puzzles")
def chess_puzzles(epochs: int = 1) -> Task:
    dataset = csv_dataset("bench/task/chess_puzzles/puzzles.csv")

    plan = [prompt_template(SOLUTION_PROMPT_TEMPLATE), generate()]

    reducers = ["mean"]

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=model_extracted_exact_match(),
        epochs=Epochs(epochs, reducers),
        metadata={
            "epochai-benchmark-version": "1.1.4",
            "inspect-log-public": True,
        },
    )


AGENT_QUESTION_TEMPLATE = (
    "Below is a FEN string representing the state of a chess game. Analyze the position and "
    "determine the best next move for the player whose turn it is.\n"
    "\n"
    "{prompt}"
)


AGENT_SUBMIT_DESCRIPTION = (
    "Submit the best next move as a four-character string stating the starting and ending squares "
    "of the piece to be moved, e.g. b1c3. The conversation will end immediately."
)


@task(name="Chess Puzzles (Agent)")
def chess_puzzles_agent(
    *,
    epochs: int = 1,
    max_output_tokens: int,
    compaction_threshold_tokens: int = 100_000,
) -> Task:
    dataset = csv_dataset("bench/task/chess_puzzles/puzzles.csv")

    plan = [
        prompt_template(AGENT_QUESTION_TEMPLATE),
        qa_agent(
            max_output_tokens=max_output_tokens,
            compaction_threshold_tokens=compaction_threshold_tokens,
            submit_description=AGENT_SUBMIT_DESCRIPTION,
        ),
    ]

    reducers = ["mean"]

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=exact(),
        epochs=Epochs(epochs, reducers),
        metadata={
            "epochai-benchmark-version": "1.0.2",
            "inspect-log-public": True,
        },
    )
