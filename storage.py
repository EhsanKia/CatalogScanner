from common import ScanMode, ScanResult

import cv2
import numpy

from typing import Iterator, List

# The expected color for the video background.
BG_COLOR = numpy.array([69, 198, 246])


def scan_storage(video_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans a video of scrolling through storage returns all items found."""
    item_images = parse_video(video_file)
    item_names = match_items(item_images)
    results = translate_names(item_names, locale)

    return ScanResult(
        mode=ScanMode.STORAGE,
        items=results,
        locale=locale.replace('auto', 'en-us'),
    )


def parse_video(filename: str) -> List[numpy.ndarray]:
    """Parses a whole video and returns images for all storage items found."""
    all_rows: List[numpy.ndarray] = []
    for i, frame in enumerate(_read_frames(filename)):
        if i % 4 != 0:
            continue  # Skip every 4th frame
        for new_row in _parse_frame(frame):
            if _is_duplicate_row(all_rows, new_row):
                continue  # Skip non-moving frames
            all_rows.extend(new_row)
    return all_rows


def match_items(item_images: List[numpy.ndarray]) -> List[str]:
    """Matches a list of names against a database of items, finding best matches."""
    # TODO: Implement image to item matching.
    return []


def translate_names(item_names: List[str], locale: str) -> List[str]:
    """Translates a list of item names to the given locale."""
    # TODO: Implement translation
    return item_names


def _read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region."""
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        assert frame.shape[:2] == (720, 1280), \
            'Invalid resolution: {1}x{0}'.format(*frame.shape)

        color = frame[:30, 1100:1150].mean(axis=(0, 1))
        if numpy.linalg.norm(color - BG_COLOR) > 5:
            continue  # Skip frames that are not showing storage.

        # Crop the region containing storage items.
        yield frame[150:675, 112:1168]
    cap.release()


def _parse_frame(frame: numpy.ndarray) -> Iterator[List[numpy.ndarray]]:
    """Parses an individual frame and extracts cards from the storage."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x_lines = list(range(0, 1057, 132))

    cols = []
    for x in x_lines[1:-1]:
        empty_col = gray[:, x-10:x+10]
        if empty_col.min() < 200:
            continue  # Skip columns with occlusions
        cols.append(empty_col)

    thresh = cv2.threshold(cv2.hconcat(cols), 236, 255, 0)[1]
    separators = thresh.mean(axis=1) < 240
    y_lines = [0] + list(separators.nonzero()[0])

    height_correction = 0
    for y1, y2 in zip(y_lines, y_lines[1:]):
        if not (118 < y2 - y1 < 130):
            height_correction += 1
            continue  # Invalid row size

        # Skip row when tooltip is overlapping the item.
        tooltip = cv2.inRange(frame[y2-10:y2-5, :], (160, 195, 80), (180, 205, 100))
        if tooltip.mean() > 10:
            continue

        y1 -= height_correction // 2
        yield [frame[y1+2:y1+124, x1+5:x2-5]
               for x1, x2 in zip(x_lines, x_lines[1:])]
        height_correction = 0


def _is_duplicate_row(all_rows: List[numpy.ndarray], new_row: List[numpy.ndarray]) -> bool:
    """Checks if the new row is the same as the previous seen rows."""
    if not new_row or len(all_rows) < len(new_row):
        return False

    new_concat = cv2.hconcat(new_row)
    # Checks the last 3 rows for similarities to the newly added row.
    for ind in [slice(-8, None), slice(-16, -8), slice(-24, -16), slice(-32, -24)]:
        old_concat = cv2.hconcat(all_rows[ind])
        if old_concat is None:
            continue
        if cv2.absdiff(new_concat, old_concat).mean() < 9:
            return True

    return False
