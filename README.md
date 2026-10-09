# Chess Puzzles

Epoch AI's Chess Puzzles benchmark.

Despite the name, this is not a live mirror: it's a one-off snapshot of our internal (private) benchmarks repository, and won't automatically track changes made there. It was made by:

1. running [`git filter-repo`](https://github.com/newren/git-filter-repo) on the internal repo to keep only the chess task and the files it depends on (`bench/task/chess_puzzles/`, `bench/task/model_extracted_answer.py`, `bench/solver/`), preserving their commit history;
2. lightly editing a few commit messages that referenced unrelated internal work;
3. adding a final commit that makes the code standalone (grader model resolved via Inspect model roles rather than our internal provider wrappers, plus `pyproject.toml`, `uv.lock` and this README).

Earlier commits import internal modules that aren't included here, so only the latest commit is runnable. Some commit messages mention other internal benchmarks and tooling that aren't part of this repo.

Each of the 100 samples in `bench/task/chess_puzzles/puzzles.csv` is a FEN position; the target is the best next move in `<from><to>` square notation (e.g. `d4d3`).

Two [Inspect](https://inspect.aisi.org.uk/) tasks are defined in `bench/task/chess_puzzles/__init__.py`:

- **Chess Puzzles**: the task behind all our published Chess Puzzles results. Single generation; a grader model extracts the final move from the response, which is then exact-matched against the target.
- **Chess Puzzles (Agent)**: an experimental variant that we don't (yet) use for any reported results. The model runs in a loop (`bench/solver/qa_agent.py`) with a `submit_answer` tool, an output-token budget and context compaction; the submitted move is exact-matched.

## Running

```sh
uv sync
uv run inspect eval 'bench/task/chess_puzzles/__init__.py@Chess Puzzles' \
    --model openai/gpt-5.5 --model-role grader=openai/gpt-5.6-terra

uv run inspect eval 'bench/task/chess_puzzles/__init__.py@Chess Puzzles (Agent)' \
    --model openai/gpt-5.5 -T max_output_tokens=1000000
```

Internally, models are called through Epoch's own provider wrappers; here they go through Inspect's standard providers, so set the usual API key environment variables (e.g. `OPENAI_API_KEY`).
