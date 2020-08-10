import dataclasses
import enum

from typing import List


class ScanMode(enum.Enum):
    CATALOG = 1
    RECIPES = 2
    STORAGE = 3
    CRITTERS = 4


@dataclasses.dataclass
class ScanResult:
    mode: ScanMode
    items: List[str]
    locale: str
