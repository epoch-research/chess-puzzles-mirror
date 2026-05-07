import bench
import bench.task.chess_puzzles
from _common import load_example_config

provider_cls, model_name, sample_id = load_example_config()

bench.eval(
    task=bench.task.chess_puzzles.chess_puzzles(),
    provider_cls=provider_cls,
    model_name=model_name,
    sample_id=sample_id,
)
