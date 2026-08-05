from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TopicDefinition:
    """A topic boundary expressed solely through positive and negative seed phrases."""

    name: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TopicDefinition":
        name = data.get("topic") or data.get("name")
        positive = data.get("positive")
        negative = data.get("negative")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Topic file needs a non-empty 'topic' field.")
        if not isinstance(positive, list) or not all(isinstance(item, str) for item in positive):
            raise ValueError("Topic file needs a 'positive' list of strings.")
        if not isinstance(negative, list) or not all(isinstance(item, str) for item in negative):
            raise ValueError("Topic file needs a 'negative' list of strings.")
        if not positive or not negative:
            raise ValueError("A topic needs at least one positive and one negative phrase.")
        return cls(name=name.strip(), positive=tuple(positive), negative=tuple(negative))

    @classmethod
    def from_file(cls, path: str | Path) -> "TopicDefinition":
        with Path(path).open(encoding="utf-8") as source:
            data = yaml.safe_load(source)
        if not isinstance(data, dict):
            raise ValueError("Topic file must contain a YAML object.")
        return cls.from_mapping(data)


@dataclass(frozen=True)
class LabeledExample:
    text: str
    labels: dict[str, bool]

    def label_for(self, topic: str) -> bool:
        try:
            return bool(self.labels[topic])
        except KeyError as error:
            raise ValueError(f"Dataset example is missing label '{topic}'.") from error
