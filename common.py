import dataclasses
import enum
import os

from typing import List

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def is_image_file(filename: str) -> bool:
    """Returns True if filename points to a static image (including %d sequences)."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


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
