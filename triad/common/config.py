"""Helpers for the per-module dataclass configs and their YAML overrides.

Every module keeps a single source of truth: a dataclass in
``triad/<module>/config.py`` whose defaults are the values named in the paper.
YAML files under ``configs/`` override any subset of the fields.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Type, TypeVar

import yaml

C = TypeVar("C")


@dataclass
class BaseConfig:
    """Base class offering YAML override loading for module configs."""

    @classmethod
    def from_yaml(cls: Type[C], path: str) -> C:
        with open(path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        return cls.from_dict(overrides)

    @classmethod
    def from_dict(cls: Type[C], overrides: dict) -> C:
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
        kwargs: dict[str, Any] = {k: v for k, v in overrides.items() if k in valid}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
