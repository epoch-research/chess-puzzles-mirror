from typing import Dict, Any

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import scorer, Score, Target, CORRECT, INCORRECT, accuracy, stderr
from inspect_ai.solver import generate, prompt_template, TaskState

from bench.model import default_grader_model

from typing import Dict, Any

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import scorer, Score, Target, CORRECT, INCORRECT, accuracy, stderr
from inspect_ai.solver import generate, prompt_template, TaskState
from inspect_ai.tool import ToolDef, ToolFunction

from bench.model import default_grader_model

SOLUTION_TEMPLATE = """Below is a FEN string representing the state of a chess game. Please analyze the position and determine the best next move for the player whose turn it is. You may think as much as you would like, but please conclude your response with the move expressed in algebraic notation, using the format "MOVE: Nxg4+" (without quotes).

{prompt}

"""

EXTRACTION_PROMPT_TEMPLATE = """**Task: Extract a student's answer to a single-move chess puzzle and submit the move that is their final answer.**

Instructions:
- Read the student's entire solution carefully to identify their final answer.
- The student is supposed to have concluded their solution with "MOVE: [answer]" where [answer] is given in algebraic notation.
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

        def submit(answer: str | None):
            pass

        submission_tool = ToolDef(
            tool=submit,
            name="submit",
            description="Submit the answer",
            parameters={"answer": "The answer"},
        )
        extraction_output = await default_grader_model().generate(
            extraction_prompt, tools=[submission_tool], tool_choice=ToolFunction(name="submit")
        )

        if (
            not extraction_output.message.tool_calls
            or "answer" not in extraction_output.message.tool_calls[0].arguments
        ):
            raise RuntimeError("Grader model did not correctly extract an answer")

        extracted_answer = extraction_output.message.tool_calls[0].arguments["answer"]

        if extracted_answer is None:
            return Score(
                value=INCORRECT,
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

    plan = [prompt_template(SOLUTION_TEMPLATE), generate()]

    reducers = ["mean"]

    return Task(
        dataset=dataset,
        plan=plan,
        scorer=model_extracted_exact_match(),
        epochs=Epochs(epochs, reducers),
        metadata={
            "epochai-benchmark-version": "1.0.0",
            "inspect-log-public": True,
        },
    )
