"""
Centralized configuration loader.
All scripts should use load_config() to access parameters.
"""

import yaml
from pathlib import Path
from functools import lru_cache


_CONFIG_PATH = Path(__file__).parent.parent / "config" / "params.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str = None) -> dict:
    """
    Load and cache configuration from params.yaml.
    
    Returns:
        dict with sections: data, training, neural_sde, simulation, bergomi, pricing, backtesting, outputs
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


