"""Configuration management using YAML files with recursive merge."""

import os
import yaml
from copy import deepcopy


class Config(dict):
    """Dictionary-like config with attribute access."""

    def __init__(self, d: dict = None):
        super().__init__()
        if d is not None:
            for k, v in d.items():
                self[k] = v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            dict.__setitem__(self, key, val)
        return val

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_config(config_path: str) -> Config:
    """Load a YAML configuration file and return a Config object."""
    with open(config_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    if d is None:
        d = {}
    return Config(d)


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override values take precedence."""
    merged = deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def build_config(experiment_config_path: str, default_config_path: str = None) -> Config:
    """Build a full config by merging experiment config with defaults."""
    if default_config_path is None:
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs", "default.yaml"
        )

    default = load_config(default_config_path)
    experiment = load_config(experiment_config_path)

    merged = merge_configs(default, experiment)
    return Config(merged)
