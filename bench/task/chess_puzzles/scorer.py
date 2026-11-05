"""Scoring a submitted chess move against a puzzle's solution, in whichever notation it's written."""

import re
from typing import Literal, NamedTuple

import chess
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    exact,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

# A move written as its from and to squares, with an optional separator between them ("e2-e4",
# "g4xf3"), promotion piece and check mark. Matched against the lowercased answer; any form other
# than plain UCI is a "uci_variant".
_SQUARES = re.compile(r"([a-h][1-8])[-x:]?([a-h][1-8])=?([qrbn])?[+#]?")

# Characters an answer can be wrapped in or end with that are never part of a move.
_WRAPPING = " \t\r\n`'\"*.!?"

# Piece letters a model might write in lowercase. "b" is also a file, so a lowercase "b" is read as
# a bishop only when the standard (pawn) reading isn't a legal move.
_LOWERCASE_PIECES = "kqrbn"

Notation = Literal["uci", "uci_variant", "san", "san_lowercase"]


class ReadMove(NamedTuple):
    uci: str
    notation: Notation


def read_move(fen: str, answer: str) -> ReadMove | None:
    """The move an answer names in the position `fen`, in UCI, or None if it names no single move.

    UCI is read without checking that the move is legal, since it's compared with the solution
    as is. SAN (including long algebraic like "Ng1f3" and castling) is read with python-chess
    against the position, so it must name exactly one legal move.
    """
    text = answer.strip(_WRAPPING)
    squares = _SQUARES.fullmatch(text.lower())
    if squares:
        uci = "".join(part or "" for part in squares.groups())
        notation: Notation = "uci" if text.lower() == uci else "uci_variant"
        return ReadMove(uci, notation)

    board = chess.Board(fen)
    move = _parse_san(board, text)
    if move is not None:
        return ReadMove(move.uci(), "san")
    if text and text[0] in _LOWERCASE_PIECES:
        move = _parse_san(board, text[0].upper() + text[1:])
        if move is not None:
            return ReadMove(move.uci(), "san_lowercase")
    return None


def _parse_san(board: chess.Board, text: str) -> chess.Move | None:
    try:
        move = board.parse_san(text)
    except ValueError:
        # Not SAN, an illegal move, or ambiguous between pieces.
        return None
    # python-chess reads "--" and "Z0" as a null move, which is never an answer.
    return move if move else None


@scorer(metrics=[mean(), stderr()], name="bench/exact")
def chess_move() -> Scorer:
    """Inspect's exact(), but also accepting the move written in SAN or with separators.

    Any answer exact() accepts scores the same. Otherwise, the answer is read as a move in the
    puzzle's position (see read_move) and compared with the solution in UCI. When the answer wasn't
    plain UCI, the score's metadata records `normalized_from` (the notation) and `normalized_move`
    (the move it was read as); the raw answer stays in `Score.answer`.

    It keeps exact()'s metrics and unqualified name, so its results row is published under the same
    label, with the same metrics, as runs scored by exact() itself. The "bench/" prefix keeps
    Inspect's lookup of the scorer named "exact" finding exact().
    """
    exact_match = exact()

    async def score(state: TaskState, target: Target) -> Score:
        result = await exact_match(state, target)
        assert result is not None
        if result.value == CORRECT:
            return result
        read = read_move(state.input_text, result.answer or "")
        if read is None or read.notation == "uci":
            return result
        return Score(
            value=CORRECT if read.uci in target.target else INCORRECT,
            answer=result.answer,
            metadata={"normalized_from": read.notation, "normalized_move": read.uci},
        )

    return score
