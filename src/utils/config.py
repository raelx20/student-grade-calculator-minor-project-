import os
from dataclasses import dataclass, field
from src.utils.constants import BASE_DIR, MODELS_DIR, DATA_RAW, DATA_PROCESSED, RANDOM_SEED, TEST_SIZE, CV_FOLDS


@dataclass
class Config:
    base_dir: str = BASE_DIR
    data_raw: str = DATA_RAW
    data_processed: str = DATA_PROCESSED
    models_dir: str = MODELS_DIR
    random_seed: int = RANDOM_SEED
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = True

    def ensure_dirs(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'data', 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'reports', 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'reports', 'metrics'), exist_ok=True)


config = Config()
