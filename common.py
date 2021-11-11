import dataclasses
import enum

from typing import List


class ScanMode(enum.Enum):
    CATALOG = 1
    RECIPES = 2
    STORAGE = 3
    CRITTERS = 4
    REACTIONS = 5
    MUSIC = 6


@dataclasses.dataclass
class ScanResult:
    mode: ScanMode
    items: List[str]
    locale: str
    unmatched: List[str] = dataclasses.field(default_factory=list)
