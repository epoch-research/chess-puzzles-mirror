# Chess Puzzles

Epoch AI's Chess Puzzles benchmark, extracted from our internal benchmarks repository (history preserved for the relevant files).

Each of the 100 samples in `bench/task/chess_puzzles/puzzles.csv` is a FEN position; the target is the best next move in `<from><to>` square notation (e.g. `d4d3`).

Two [Inspect](https://inspect.aisi.org.uk/) tasks are defined in `bench/task/chess_puzzles/__init__.py`:

- **Chess Puzzles**: single generation; a grader model extracts the final move from the response, which is then exact-matched against the target.
- **Chess Puzzles (Agent)**: the model runs in a loop (`bench/solver/qa_agent.py`) with a `submit_answer` tool, an output-token budget and context compaction; the submitted move is exact-matched.

## Running

```sh
uv sync
uv run inspect eval 'bench/task/chess_puzzles/__init__.py@Chess Puzzles' \
    --model openai/gpt-5.5 --model-role grader=openai/gpt-5.6-terra

uv run inspect eval 'bench/task/chess_puzzles/__init__.py@Chess Puzzles (Agent)' \
    --model openai/gpt-5.5 -T max_output_tokens=1000000
```

Internally, models are called through Epoch's own provider wrappers; here they go through Inspect's standard providers, so set the usual API key environment variables (e.g. `OPENAI_API_KEY`).
