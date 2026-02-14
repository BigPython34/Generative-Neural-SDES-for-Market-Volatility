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


def get_data_config() -> dict:
    """Shortcut for data section."""
    return load_config()['data']


def get_pricing_config() -> dict:
    """Shortcut for pricing section."""
    return load_config()['pricing']


def get_bergomi_config() -> dict:
    """Shortcut for bergomi section."""
    return load_config()['bergomi']


def get_simulation_config() -> dict:
    """Shortcut for simulation section."""
    return load_config()['simulation']


def get_backtesting_config() -> dict:
    """Shortcut for backtesting section."""
    return load_config()['backtesting']
