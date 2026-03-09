from dotenv import load_dotenv
from environs import Env
import importlib

import bench.task.chess_puzzles

load_dotenv()
env = Env()

# Get provider class from environment variable
provider_cls_path = env.str("EXAMPLE_PROVIDER_CLS")
module_path, class_name = provider_cls_path.rsplit(".", 1)
module = importlib.import_module(module_path)
provider_cls = getattr(module, class_name)

# Get model name from environment variable
model_name = env.str("EXAMPLE_MODEL_NAME")

bench.eval(
    task=bench.task.chess_puzzles.chess_puzzles(),
    provider_cls=provider_cls,
    model_name=model_name,
)
