import itertools
from common import ScanMode, ScanResult

import collections
import cv2
import enum
import functools
import json
import numpy
import os

from typing import Dict, Iterator, List, Tuple

# The expected color for the video background.
BG_COLOR = numpy.array([207, 238, 240])


class CritterType(enum.Enum):
    INSECTS = 1
    FISH = 2
    SEA_CREATURES = 3

    @classmethod
    def from_str(cls, value: str) -> 'CritterType':
        key = value.upper().replace(' ', '_')
        return cls.__members__[key]


class CritterIcon:
    """The image and data associated with a critter icon."""

    def __init__(self, critter_name: str, icon_name: str, critter_type: CritterType):
        img_path = os.path.join('critters', 'generated', icon_name)
        self.img = cv2.imread(img_path)
        self.critter_name = critter_name
        self.critter_type = critter_type

    def __repr__(self):
        return f'CritterIcon({self.item_name!r}, {self.icon_name!r}, {self.critter_type!r})'


def scan_critters(video_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans a video of scrolling through Critterpedia and returns all critters found."""
    critter_icons = parse_video(video_file)
    critter_names = match_critters(critter_icons)
    results = translate_names(critter_names, locale)

    return ScanResult(
        mode=ScanMode.CRITTERS,
        items=results,
        locale=locale.replace('auto', 'en-us'),
    )


def parse_video(filename: str) -> List[Tuple[CritterType, numpy.ndarray]]:
    """Parses a whole video and returns icons for all critters found."""
    icons: Dict[CritterType, List[numpy.ndarray]] = collections.defaultdict(list)
    for i, (critter_type, frame) in enumerate(_read_frames(filename)):
        for new_icons in _parse_frame(frame):
            icons[critter_type].extend(new_icons)

    all_icons: List[Tuple[CritterType, numpy.ndarray]] = []
    for critter_type, critter_icons in icons.items():
        all_icons.extend((critter_type, i) for i in _remove_blanks(critter_icons))
    return all_icons


def match_critters(critter_icons: List[Tuple[CritterType, numpy.ndarray]]) -> List[str]:
    """Matches a list of names against a database of items, finding best matches."""
    matched_critters = set()
    critter_db = _get_critter_db()
    for critter_type, critter_icon in critter_icons:
        best_match = _find_best_match(critter_icon, critter_db[critter_type])
        matched_critters.add(best_match.critter_name)
    return list(matched_critters)


def translate_names(critter_names: List[str], locale: str) -> List[str]:
    """Translates a list of critter names to the given locale."""
    if locale in ['auto', 'en-us']:
        return critter_names

    translation_path = os.path.join('critters', 'translations.json')
    with open(translation_path, encoding='utf-8') as fp:
        translations = json.load(fp)
    return [translations[name][locale] for name in critter_names]


def _read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region."""
    frame_skip = 0
    last_section = None
    last_gray = None

    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        if frame_skip > 0:
            frame_skip -= 1
            continue

        assert frame.shape[:2] == (720, 1280), \
            'Invalid resolution: {1}x{0}'.format(*frame.shape)

        color = frame[:20, 1100:1150].mean(axis=(0, 1))
        if numpy.linalg.norm(color - BG_COLOR) > 5:
            continue  # Skip frames that are not showing critterpedia.

        # Detect a dark line that shows up only in Pictures Mode.
        mode_detector = frame[20:24, 600:800].mean(axis=(0, 1))
        if numpy.linalg.norm(mode_detector - (199, 234, 237)) > 50:
            raise AssertionError('Critterpedia is in Pictures Mode.')

        # Only parse frames that are at the very left or very right of the list.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray[150:620, :30].min() < 200 and gray[150:620, -30:].min() < 200:
            continue

        # Skip few frames after section changes to allow icons to load.
        critter_section = _detect_critter_type(gray)
        if critter_section != last_section:
            if last_section is not None:
                frame_skip = 15
            last_section = critter_section
            continue

        # Skip non-moving frames.
        if last_gray is not None and cv2.absdiff(gray, last_gray).mean() < 4:
            continue
        last_gray = gray

        # Skip empty frames.
        if gray[149:623, :].min() > 5:
            continue

        # Get the section name and crop the region containing critter icons.
        yield critter_section, frame[149:623, :]
    cap.release()


def _detect_critter_type(gray_frame: numpy.ndarray) -> CritterType:
    for i, critter_type in enumerate(CritterType):
        start_x, end_x = 65 + i * 65, 65 + (i + 1) * 65
        section_icon = gray_frame[50:100, start_x:end_x]
        if section_icon.min() > 150:
            return critter_type
    raise AssertionError('Invalid Critterpedia page')


def _parse_frame(frame: numpy.ndarray) -> Iterator[List[numpy.ndarray]]:
    """Parses an individual frame and extracts icons from the Critterpedia page."""
    # Start/end verical position for the 5 grid rows.
    y_positions = [0, 95, 190, 285, 379]
    y_offsets = [6, 88]

    rows = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for y_pos, offset in itertools.product(y_positions, y_offsets):
        line = gray[y_pos + offset - 3:y_pos + offset + 3, :]
        if line.min() < 170 or line.max() > 240:
            continue
        rows.append(line)

    if not rows:
        return

    thresh = cv2.threshold(cv2.vconcat(rows), 215, 255, 0)[1]
    separators = thresh.mean(axis=0) < 240
    x_lines = list(separators.nonzero()[0])

    for x1, x2 in zip(x_lines, x_lines[1:]):
        if not (107 < x2 - x1 < 117):
            continue
        yield [frame[y+8:y+88, x1+16:x1+96] for y in y_positions]


def _remove_blanks(all_icons: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """Remove all icons that show empty critter boxes."""
    filtered_icons = []
    for icon in all_icons:
        if icon[20:60, 20:60, 2].min() > 100:
            continue
        filtered_icons.append(icon)
    return filtered_icons


@functools.lru_cache()
def _get_critter_db() -> Dict[CritterType, List[CritterIcon]]:
    """Fetches the item database for a given locale, with caching."""
    with open(os.path.join('critters', 'names.json')) as fp:
        critter_data = json.load(fp)

    critter_db = collections.defaultdict(list)
    for critter_name, icon_name, critter_type_str in critter_data:
        critter_type = CritterType.from_str(critter_type_str)
        critter = CritterIcon(critter_name, icon_name, critter_type)
        critter_db[critter_type].append(critter)
    return critter_db


def _find_best_match(icon: numpy.ndarray, critters: List[CritterIcon]) -> CritterIcon:
    """Finds the closest matching critter for the given icon."""
    fast_similarity_metric = lambda r: cv2.absdiff(icon, r.img).mean()
    similarities = list(map(fast_similarity_metric, critters))
    sim1, sim2 = numpy.partition(similarities, kth=2)[:2]

    # If the match seems obvious, return the quick result.
    if abs(sim1 - sim2) > 3:
        return critters[numpy.argmin(similarities)]

    # Otherwise, we use a slower matching, which tries various shifts.
    def slow_similarity_metric(critter):
        diffs = []
        for x in [-2, -1, 0, 1, 2]:
            shifted = numpy.roll(icon, x, axis=1)
            diffs.append(cv2.absdiff(shifted, critter.img).sum())
        return min(diffs)  # Return lowest diff across shifts.

    similarities = list(map(slow_similarity_metric, critters))
    return critters[numpy.argmin(similarities)]


if __name__ == "__main__":
    results = scan_critters('videos/critters.mp4')
    print(len(results.items))
