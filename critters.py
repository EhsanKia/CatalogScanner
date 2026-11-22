import itertools
from common import ScanMode, ScanResult, crop_720p_roi, is_image_file, slice_720p

import collections
import cv2
import enum
import functools
import json
import logging
import numpy
import os

from typing import Dict, Iterator, List, Tuple

# The expected color for the video background.
BG_COLOR = numpy.array([207, 238, 240])
# Blurred slow-scroll icons (e.g. dace in critters_slow.mp4) can reach ~36 L1; noise/blank boxes are >75.
MAX_MATCH_DISTANCE = 42.0
COARSE_TOP_K = 12


class CritterType(enum.Enum):
    INSECTS = 1
    FISH = 2
    SEA_CREATURES = 3

    @classmethod
    def from_str(cls, value: str) -> 'CritterType':
        key = value.upper().replace(' ', '_')
        return cls.__members__[key]


class CritterImage:
    """The image and data associated with a critter icon."""

    def __init__(self, critter_name: str, critter_type: CritterType, icon_name: str):
        img_path = os.path.join('critters', 'generated', icon_name)
        self.img = cv2.imread(img_path)
        self.critter_name = critter_name
        self.critter_type = critter_type
        self.icon_name = icon_name

    def __repr__(self):
        return f'CritterIcon({self.critter_name!r}, {self.critter_type!r}, {self.icon_name!r})'


class CritterIcon(numpy.ndarray):
    """Dummy ndarray subclass to hold critter type info."""
    critter_type: CritterType


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing Critterpedia."""
    color = slice_720p(frame, None, 20, 1100, 1150).mean(axis=(0, 1))
    return bool(numpy.linalg.norm(color - BG_COLOR) < 12)


def scan(video_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans a video of scrolling through Critterpedia and returns all critters found."""
    critter_icons = parse_video(video_file)
    critter_names, unmatched = match_critters(critter_icons)
    results = translate_names(critter_names, locale)

    return ScanResult(
        mode=ScanMode.CRITTERS,
        items=results,
        locale=locale.replace('auto', 'en-us'),
        unmatched=unmatched,
    )


def parse_video(filename: str) -> List[CritterIcon]:
    """Parses a whole video and returns icons for all critters found."""
    all_icons: List[CritterIcon] = []
    section_count: Dict[CritterType, int] = collections.defaultdict(int)
    for critter_type, frame in _read_frames(filename):
        section_count[critter_type] += 1
        for new_icon in _parse_frame(frame):
            critter_icon = new_icon.view(CritterIcon)
            critter_icon.critter_type = critter_type
            all_icons.append(critter_icon)

    assert section_count[CritterType.INSECTS] != 1, \
        'Incomplete critter scan for INSECTS section.'
    assert section_count[CritterType.FISH] != 1, \
        'Incomplete critter scan for FISH section.'

    return _remove_blanks(all_icons)


def match_critters(critter_icons: List[CritterIcon]) -> Tuple[List[str], List[str]]:
    """Matches icons against database of critter images, finding best matches."""
    matched_critters = set()
    unmatched: List[str] = []
    if not critter_icons:
        return [], unmatched

    coarse_db = _get_critter_coarse_db()
    icon_vecs = numpy.stack([
        cv2.resize(ic, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        for ic in critter_icons
    ]).astype(numpy.float32)

    for icon, icon_vec in zip(critter_icons, icon_vecs):
        clist, templates, norms = coarse_db[icon.critter_type]
        if len(clist) > COARSE_TOP_K:
            d2 = norms - 2.0 * (templates @ icon_vec)
            top_idx = numpy.argpartition(d2, COARSE_TOP_K - 1)[:COARSE_TOP_K]
            candidates = [clist[int(i)] for i in top_idx]
        else:
            candidates = clist

        best_match, best_dist = _find_best_match(icon, candidates)
        if best_dist >= MAX_MATCH_DISTANCE:
            logging.warning(
                'Rejected critter icon with L1 distance %.1f (closest: %s)',
                best_dist, best_match.critter_name,
            )
            unmatched.append(f'{best_match.critter_name}?d={best_dist:.1f}')
            continue
        matched_critters.add(best_match.critter_name)
    return sorted(matched_critters), unmatched


def translate_names(critter_names: List[str], locale: str) -> List[str]:
    """Translates a list of critter names to the given locale."""
    if locale in ['auto', 'en-us']:
        return critter_names

    translation_path = os.path.join('critters', 'translations.json')
    with open(translation_path, encoding='utf-8') as fp:
        translations = json.load(fp)
    return [translations[name][locale] for name in critter_names]


def _read_frames(filename: str) -> Iterator[Tuple[CritterType, numpy.ndarray]]:
    """Parses frames of the given video and returns the relevant region."""
    frame_skip = 0
    last_section = None
    last_frame = None
    good_frames: Dict[Tuple[CritterType, int], numpy.ndarray] = {}

    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    section_skip = max(15, round(15 * fps / 30.0)) if fps > 0 else 15
    try:
        while True:
            if frame_skip > 0:
                if not cap.grab():
                    break
                frame_skip -= 1
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                break  # Video is over

            if not detect(frame):
                continue  # Skip frames that are not showing critterpedia.

            # Detect a dark line that shows up only in Pictures Mode.
            mode_detector = slice_720p(frame, 20, 24, 600, 800).mean(axis=(0, 1))
            if numpy.linalg.norm(mode_detector - (199, 234, 237)) > 50:
                raise AssertionError('Critterpedia is in Pictures Mode.')

            tab_gray = cv2.cvtColor(crop_720p_roi(frame, 70, 75, 65, 260), cv2.COLOR_BGR2GRAY)
            critter_section = _detect_critter_section(tab_gray)

            if is_image_file(filename):  # Handle screenshots
                yield critter_section, crop_720p_roi(frame, 149, 623, None, None)
                continue

            if last_frame is None:
                last_frame = frame
                continue

            if critter_section != last_section:
                if last_section is not None:
                    frame_skip = section_skip
                last_section = critter_section
                continue

            # Grab the last frame for each side and section combination.
            if slice_720p(last_frame, 570, 600, None, 70)[:, :, 2].min() > 230:
                good_frames[critter_section, 0] = last_frame
            elif slice_720p(last_frame, 570, 600, -70, None)[:, :, 2].min() > 230:
                good_frames[critter_section, 1] = last_frame

            last_frame = frame
    finally:
        cap.release()

    for (critter_type, _), frame in good_frames.items():
        # Crop the region containing critter icons.
        yield critter_type, crop_720p_roi(frame, 149, 623, None, None)


def _detect_critter_section(tab_gray: numpy.ndarray) -> CritterType:
    for i, critter_type in enumerate(CritterType):
        section_icon = tab_gray[:, i * 65:(i + 1) * 65]
        if section_icon.min() > 150:
            return critter_type
    raise AssertionError('Invalid Critterpedia page')


def _parse_frame(frame: numpy.ndarray) -> Iterator[numpy.ndarray]:
    """Parses an individual frame and extracts icons from the Critterpedia page."""
    # Start/end verical position for the 5 grid rows.
    y_positions = [0, 95, 190, 285, 379]
    y_offsets = [5, 89]

    rows = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for y_pos, offset in itertools.product(y_positions, y_offsets):
        line = gray[y_pos + offset - 3:y_pos + offset + 3, :]
        if line.min() < 170 or line.max() > 240:
            continue
        rows.append(line)

    if not rows:
        return

    thresh = cv2.threshold(cv2.vconcat(rows), 210, 255, 0)[1]
    separators = thresh.mean(axis=0) < 240
    x_lines = list(separators.nonzero()[0])

    # Normalize column lines by taking the average of all of them.
    # We know they are 112.7px apart, so we find the best offset from given lines.
    centers = [numpy.fmod(x, 112.7) for x in x_lines]
    centroid = round(numpy.median(centers))
    x_positions = numpy.arange(centroid, 1280, 112.7).astype(int)

    for x, y in itertools.product(x_positions, y_positions):
        if x + 96 > frame.shape[1]:
            continue  # Past the right side of the frame
        yield frame[y+8:y+88, x+16:x+96]


def _remove_blanks(all_icons: List[CritterIcon]) -> List[CritterIcon]:
    """Remove all icons that show empty critter boxes."""
    filtered_icons = []
    for icon in all_icons:
        if icon[20:60, 20:60, 2].min() > 100:
            continue
        filtered_icons.append(icon)
    return filtered_icons


@functools.lru_cache()
def _get_critter_db() -> Dict[CritterType, List[CritterImage]]:
    """Fetches the critters database for a given locale, with caching."""
    with open(os.path.join('critters', 'names.json')) as fp:
        critter_data = json.load(fp)

    critter_db = collections.defaultdict(list)
    for critter_name, icon_name, critter_type_str in critter_data:
        critter_type = CritterType.from_str(critter_type_str)
        critter = CritterImage(critter_name, critter_type, icon_name)
        critter_db[critter_type].append(critter)
    return critter_db


@functools.lru_cache()
def _get_critter_coarse_db() -> Dict[CritterType, Tuple[List[CritterImage], numpy.ndarray, numpy.ndarray]]:
    """Precomputes 16x16 flattened float32 templates and squared norms per CritterType."""
    coarse_db: Dict[CritterType, Tuple[List[CritterImage], numpy.ndarray, numpy.ndarray]] = {}
    for critter_type, critter_list in _get_critter_db().items():
        small = numpy.stack([
            cv2.resize(c.img, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
            for c in critter_list
        ]).astype(numpy.float32)
        norms = (small * small).sum(axis=1)
        coarse_db[critter_type] = (critter_list, small, norms)
    return coarse_db


def _find_best_match(icon: numpy.ndarray, candidates: List[CritterImage]) -> Tuple[CritterImage, float]:
    """Finds the closest matching critter and its mean L1 pixel distance."""
    sims = [float(cv2.absdiff(icon, c.img).mean()) for c in candidates]
    if len(candidates) == 1 and sims[0] < MAX_MATCH_DISTANCE:
        return candidates[0], sims[0]

    if len(candidates) > 1:
        order = numpy.argsort(sims)
        if sims[int(order[1])] - sims[int(order[0])] > 3 and sims[int(order[0])] < MAX_MATCH_DISTANCE:
            best_idx = int(order[0])
            return candidates[best_idx], sims[best_idx]

    # Try small horizontal shifts when the match is ambiguous or above threshold.
    shifted_icons = [numpy.roll(icon, x, axis=1) for x in (-2, -1, 0, 1, 2)]
    best_dists = [min(float(cv2.absdiff(si, c.img).mean()) for si in shifted_icons) for c in candidates]
    best_idx = int(numpy.argmin(best_dists))
    return candidates[best_idx], best_dists[best_idx]


if __name__ == "__main__":
    results = scan('examples/critters.mp4')
    print('\n'.join(results.items))
