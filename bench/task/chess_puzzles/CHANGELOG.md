# Changelog

Dates are when the version reached `main`.

Some settings are not part of a version and can differ between runs of the same version: the
reasoning effort, and in 1.x the per-response output-token limit (`max_tokens`, set by the run,
otherwise the model's own maximum). In 2.0.0 the output-token budget
(`max_output_tokens`, 1M in hawkbench's defaults) and the compaction threshold (90% of the model's
context window in hawkbench's defaults) are task arguments, recorded in each eval log. The LLM
grader used in 1.x was hard-coded rather than a run setting, but changed once without a version
bump (see 1.1.6).

## 2.0.1 (2026-10-14)

- An OpenAI response cut off while still reasoning, with encrypted reasoning too long for OpenAI to
  accept back (over 1,048,576 characters, which a 128k-token response exceeds), now gets the
  forced final answer, sent without that reasoning. Before, continuing it failed with a 400
  (`string_above_max_length`) and the sample errored. Recorded as `forced_final_answer`, like a
  budget run-out. Other providers and other samples are unchanged.

## 2.0.0 (2026-10-13)

Scores are not comparable with 1.x.

- The task runs on `qa_agent` (the same harness as Mystery Game Puzzles) instead of a single call.
  The model answers by calling a `submit_answer` tool, within an output-token budget (1M in
  hawkbench's defaults).
  - A response cut off at the model's own output-token limit (128k for most models) is now
    continued; in 1.x it was scored as no answer. This mostly affected max-effort configs (e.g.
    Opus 4.7 max had 88/100 samples cut off).
  - If the budget runs out before a submission, the model is asked for an immediate answer at its
    lowest reasoning effort (recorded in sample metadata as `forced_final_answer`).
  - On Zhipu and Alibaba models that support it, the model keeps its earlier turns' reasoning
    (`preserve_thinking`).
- The submitted move is scored by exact match, replacing the LLM answer extraction (and its
  gpt-5-mini grader).
- The prompt changed: the one-call "conclude with `MOVE: b1c3`" instruction is replaced by
  `qa_agent`'s wrapper (which states the token budget) and a `submit_answer` tool description.
  - Configs with little or no reasoning (e.g. `minimal` effort) score lower than in 1.x. With a
    `submit_answer` tool they answer almost immediately, while the 1.x prompt ("You may think as
    much as you would like") led them to write out their analysis before the move. For example,
    Gemini 3.5 Flash at `minimal` effort went from 43% to 27%. Configs that reason at medium effort
    or more score about the same as in 1.x, apart from the cut-off fix above.
- Samples are numbered 1–100 in dataset order, so they can be selected by id.
- `max_output_tokens` and `compaction_threshold_tokens` are required task arguments, with no
  defaults in the task.

## 1.1.6 (2026-06-11)

- Dependency update (inspect-ai 0.3.239). No change to the task.
- 2026-08-10, without a version bump: the answer-extraction grader changed from gpt-5-mini to
  gpt-5.6-terra (medium reasoning effort) for every task using the shared default grader (#384).
  1.1.6 runs before that date were graded by gpt-5-mini, later ones by gpt-5.6-terra.

## 1.1.5 (2026-06-04)

- Python 3.13 upgrade. No change to the task.

## 1.1.4 (2026-05-06)

- Dependency update (inspect-ai 0.3.218). No change to the task.
- 2026-05-11, without a version bump: the answer-extraction tool's parameters were made explicit
  (shared `model_extracted_answer` helper). Extraction behavior is unchanged.

## 1.1.3 (2026-04-24)

- Dependency update (inspect-ai 0.3.211).
- Restores the version number: on 2026-04-16 a revert (#337) set it back to 1.0.1 by mistake, so
  runs from 2026-04-16 to 2026-04-24 are tagged 1.0.1 but ran 1.1.1's task (with the answer
  extraction moved into a shared helper on 2026-04-15, #336). 1.1.2 was never used.

## 1.1.1 (2026-03-09)

- The answer-extraction grader changed from gpt-5-nano to gpt-5-mini (medium reasoning effort),
  which was more accurate in manual grading.

## 1.1.0 (2026-02-25)

- The answer-extraction grader changed from gemini-2.0-flash (being deprecated) to gpt-5-nano.
- The scorer treats a grader answer of `"None"` (the string) as no answer, like `None`.

## 1.0.2 (2026-02-12)

- Dependency update (inspect-ai 0.3.174, with support for Opus 4.6 adaptive-thinking effort). No
  change to the task.

## 1.0.1 (2026-01-20)

- Dependency update (inspect-ai 0.3.162). No change to the task.

## 1.0.0 (2025-12-10)

- Initial version: 100 single-move puzzles given as FEN strings. The model answers in one call,
  concluding with the move in the form `MOVE: b1c3`; gemini-2.0-flash extracts the move, which is
  scored by exact match against the solution.
