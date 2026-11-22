import dataclasses
import enum
import logging
import os

import cv2
import numpy

from typing import Callable, Iterator, List, Optional, Tuple, Union

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def is_image_file(filename: str) -> bool:
    """Returns True if filename points to a static image (including %d sequences)."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def to_720p(frame: numpy.ndarray) -> numpy.ndarray:
    """Resizes frame to 720p (1280x720) if needed."""
    if frame.shape[:2] != (720, 1280):
        return cv2.resize(frame, (1280, 720))
    return frame


def iter_frames(
    source: Union[str, cv2.VideoCapture],
    detect_fn: Callable[[numpy.ndarray], bool],
    log_label: Optional[str] = None,
) -> Iterator[Tuple[numpy.ndarray, int]]:
    """Yields 720p frames matching detect_fn alongside the mid-video gap count before each frame."""
    cap = cv2.VideoCapture(source) if isinstance(source, str) else source
    label = log_label or (source if isinstance(source, str) else '<video>')
    seen_valid = False
    pending_gap = 0
    total_gap_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = to_720p(frame)
            try:
                is_valid = detect_fn(frame)
            except AssertionError:
                if not seen_valid:
                    raise
                is_valid = False

            if not is_valid:
                if seen_valid:
                    pending_gap += 1
                continue

            gap_before = pending_gap
            if pending_gap:
                total_gap_frames += pending_gap
                pending_gap = 0
            seen_valid = True
            yield frame, gap_before
    finally:
        cap.release()
        if total_gap_frames:
            logging.warning('Skipped %d non-matching mid-video frames in %s', total_gap_frames, label)


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
