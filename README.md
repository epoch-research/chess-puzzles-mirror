# Chess Puzzles

Epoch AI's Chess Puzzles benchmark.

Despite the name, this is not a live mirror: it's a snapshot of our internal (private) benchmarks repository, refreshed by hand, and doesn't automatically track changes made there. It was made by:

1. running [`git filter-repo`](https://github.com/newren/git-filter-repo) on the internal repo to keep only the chess task and the files it depends on (`bench/task/chess_puzzles/`, `bench/task/model_extracted_answer.py`, `bench/solver/`), preserving their commit history;
2. lightly editing a few commit messages that referenced unrelated internal work;
3. adding a final commit that makes the code standalone (grader model resolved via Inspect model roles rather than our internal provider wrappers, plus `pyproject.toml`, `uv.lock` and this README).

It was refreshed on 2026-10-14, for version 2.0.1 of the task, and again on 2026-10-26 to add the Decision variant, each time by repeating step 1 on the internal repo and appending the new commits, then adding a commit that makes the new code standalone (see below).

Earlier commits import internal modules that aren't included here, so only the latest commit is runnable. Some commit messages mention other internal benchmarks and tooling that aren't part of this repo.

Each of the 100 samples in `bench/task/chess_puzzles/puzzles.csv` is a FEN position; the target is the best next move in `<from><to>` square notation (e.g. `d4d3`).

The task, **Chess Puzzles**, is defined in `bench/task/chess_puzzles/__init__.py`, and its version history is in `bench/task/chess_puzzles/CHANGELOG.md`. Since version 2.0.0, the model runs in a loop (`bench/solver/qa_agent.py`) with a `submit_answer` tool, an output-token budget and context compaction, and the submitted move is exact-matched against the target. A response cut off at the model's output-token limit is continued; if the budget runs out first, the model is asked for an immediate answer.

A second task, **Chess Puzzles (Decision)**, is for models that can't write text: the same puzzles, prompt and `submit_answer` tool, answered in one call that picks from the position's legal moves (`bench/solver/structured_answer.py`), scored by exact match. It takes optional `epochs` and `shuffle` arguments.

## Running

```sh
uv sync
uv run inspect eval 'bench/task/chess_puzzles/__init__.py@Chess Puzzles' \
    --model openai/gpt-5.5 \
    -T max_output_tokens=1000000 -T compaction_threshold_tokens=945000
```

Both task arguments are required. Our runs use a budget of 1,000,000 output tokens and a compaction threshold of 90% of the model's context window (945,000 for a 1.05M-token window).

Internally, models are called through Epoch's own provider wrappers; here they go through Inspect's standard providers, so set the usual API key environment variables (e.g. `OPENAI_API_KEY`). One behavior differs as a result: when the budget runs out, our runs make the forced final call at the model's lowest reasoning effort, read from our internal model config. Here it uses the model's default effort (see `bench/solver/forced_answer.py`).
