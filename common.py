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


def _resolve_axis_bounds(
    start: Optional[int],
    end: Optional[int],
    ref_size: int,
    actual_size: int,
) -> Tuple[int, int, int]:
    """Maps 720p-relative slice bounds to [r1, r2) on an axis and returns its 720p span."""
    ref_start = 0 if start is None else (start if start >= 0 else ref_size + start)
    ref_end = ref_size if end is None else (end if end >= 0 else ref_size + end)
    scale = actual_size / float(ref_size)

    def _map_bound(val: Optional[int], default: int) -> int:
        if val is None:
            return default
        scaled = round(val * scale)
        return scaled if val >= 0 else actual_size + scaled

    r1 = max(0, min(actual_size - 1, _map_bound(start, 0)))
    r2 = max(r1 + 1, min(actual_size, _map_bound(end, actual_size)))
    return r1, r2, max(1, ref_end - ref_start)


def slice_720p(
    frame: numpy.ndarray,
    y1: Optional[int],
    y2: Optional[int],
    x1: Optional[int],
    x2: Optional[int],
) -> numpy.ndarray:
    """Slices a 720p-defined ROI from a frame at its native resolution without full-frame resize."""
    if frame.shape[:2] == (720, 1280):
        return frame[y1:y2, x1:x2]
    h, w = frame.shape[:2]
    ry1, ry2, _ = _resolve_axis_bounds(y1, y2, 720, h)
    rx1, rx2, _ = _resolve_axis_bounds(x1, x2, 1280, w)
    return frame[ry1:ry2, rx1:rx2]


def crop_720p_roi(
    frame: numpy.ndarray,
    y1: Optional[int],
    y2: Optional[int],
    x1: Optional[int],
    x2: Optional[int],
) -> numpy.ndarray:
    """Extracts a 720p-defined ROI, resizing only the cropped region when frame is not 720p."""
    if frame.shape[:2] == (720, 1280):
        return frame[y1:y2, x1:x2]
    h, w = frame.shape[:2]
    ry1, ry2, target_h = _resolve_axis_bounds(y1, y2, 720, h)
    rx1, rx2, target_w = _resolve_axis_bounds(x1, x2, 1280, w)
    return cv2.resize(frame[ry1:ry2, rx1:rx2], (target_w, target_h))


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
