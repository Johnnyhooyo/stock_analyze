"""
统一配置加载
============
全项目唯一的配置加载入口，加载 config.yaml 和 keys.yaml 并合并。
"""

from pathlib import Path

import yaml


def load_config(include_keys: bool = True) -> dict:
    """
    统一配置加载入口。

    - config.yaml: 业务配置
    - keys.yaml:   密钥（可选，include_keys=True 时加载并合并）

    Returns:
        dict: 合并后的配置字典
    """
    base = Path(__file__).parent
    config = {}

    cfg_path = base / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if include_keys:
        keys_path = base / "keys.yaml"
        if keys_path.exists():
            with open(keys_path, encoding="utf-8") as f:
                keys = yaml.safe_load(f) or {}
                config.update(keys)

    return config
