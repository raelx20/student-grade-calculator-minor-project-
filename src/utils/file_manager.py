import os
import glob as glob_module
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_models(models_dir: str) -> dict:
    result = {}
    for f in glob_module.glob(os.path.join(models_dir, '*.pkl')):
        name = os.path.basename(f)
        size = os.path.getsize(f)
        result[name] = {'path': f, 'size_bytes': size}
    return result
