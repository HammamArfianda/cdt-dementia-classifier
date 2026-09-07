from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent directory for an output file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
