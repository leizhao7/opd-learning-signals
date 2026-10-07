"""Checkpoint directory preparation; no cluster-specific directory assertion."""
from pathlib import Path
import shutil
import os
def prepare(trainer):
    path = Path(trainer.config.trainer.default_local_dir)
    path.mkdir(parents=True, exist_ok=True)
    minimum = float(os.environ.get("OPD_MIN_FREE_GB", "40")) * 1024**3
    if shutil.disk_usage(path).free < minimum:
        raise RuntimeError("Insufficient checkpoint storage; adjust OPD_MIN_FREE_GB if appropriate")
