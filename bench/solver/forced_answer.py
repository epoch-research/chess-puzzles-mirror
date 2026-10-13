"""Settings for a final call that asks a model to answer now, without further reasoning."""

from bench.model.epoch.config import try_parse_epoch_model
from bench.reasoning import REASONING_EFFORTS, ReasoningEffort


def lowest_reasoning_effort(model_name: str) -> ReasoningEffort | None:
    """`none` when the model supports it, otherwise its lowest reasoning effort.

    `None` when the model declares no reasoning efforts or isn't an Epoch model.
    """
    epoch_model = try_parse_epoch_model(model_name)
    if epoch_model is None or not epoch_model.config.thinking_settings:
        return None
    settings = epoch_model.config.thinking_settings
    return next(effort for effort in REASONING_EFFORTS if effort in settings)


def supports_forced_tool_choice(model_name: str) -> bool:
    """Whether a tool can be forced via `tool_choice`.

    Non-Epoch models are assumed to support it; only an Epoch model that opts out via models.yaml
    (`supports_tool_choice: false`) doesn't.
    """
    epoch_model = try_parse_epoch_model(model_name)
    return epoch_model is None or epoch_model.supports_tool_choice
