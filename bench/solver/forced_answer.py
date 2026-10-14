"""Settings for a final call that asks a model to answer now, without further reasoning.

Internally these come from each model's entry in Epoch's model config (its supported reasoning
efforts, and whether it accepts a forced `tool_choice`). That config isn't part of this repo, so
every model is treated the way the internal code treats a model it has no entry for.
"""


def lowest_reasoning_effort(model_name: str) -> None:
    """No reasoning-effort override: the forced call uses the model's default."""
    return None


def supports_forced_tool_choice(model_name: str) -> bool:
    """Assume `submit_answer` can be forced via `tool_choice`."""
    return True
