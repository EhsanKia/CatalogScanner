from common import ScanMode, ScanResult, color_match, is_image_file, slice_720p, to_720p

import cv2
import itertools
import functools
import json
import logging
import numpy
import os

from typing import Dict, Iterator, List, Optional, Tuple

# The expected color for the reactions background.
BG_COLOR = (254, 221, 244)
MAX_MATCH_DISTANCE = 48.0
COARSE_TOP_K = 10

# The expected color for the reactions background.
BG_COLOR = (254, 221, 244)

# The color of the middle dot on empty icons.
SELECT_COLOR = (166, 190, 7)
EMPTY_COLOR = (237, 198, 215)

# The position for all 44 reaction slots, listed manually.
REACTION_POSITIONS = [
    (318, 166), (382, 141), (446, 126), (511, 116), (577, 110), (640, 108), (703, 110), (769, 116), (834, 126),
    (898, 141), (962, 166), (315, 228), (381, 207), (447, 190), (512, 179), (575, 175), (640, 174), (705, 175),
    (768, 179), (833, 190), (899, 207), (966, 228), (320, 291), (382, 269), (448, 255), (512, 244), (575, 239),
    (640, 237), (705, 239), (768, 244), (832, 255), (898, 269), (961, 291), (330, 353), (390, 334), (451, 320),
    (513, 311), (577, 304), (640, 302), (703, 304), (766, 311), (829, 320), (890, 334), (950, 353),
]


class ReactionImage:
    """The image and data associated with a reaction icon."""

    def __init__(self, reaction_name: str, filename: str):
        img_path = os.path.join('reactions', 'generated', filename)
        self.img = cv2.imread(img_path)
        self.reaction_name = reaction_name
        self.filename = filename

    def __repr__(self):
        return f'ReactionImage({self.reaction_name!r}, {self.filename!r})'


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing reactions list."""
    color = slice_720p(frame, 370, 380, 290, 300).mean(axis=(0, 1))
    return color_match(color, BG_COLOR, 14)


def scan(image_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans an image of reactions list and returns all reactions found."""
    reaction_icons = parse_image(image_file)
    reaction_names, unmatched = match_reactions(reaction_icons)
    results = translate_names(reaction_names, locale)

    return ScanResult(
        mode=ScanMode.REACTIONS,
        items=results,
        locale=locale.replace('auto', 'en-us'),
        unmatched=unmatched,
    )


def parse_image(filename: str) -> List[numpy.ndarray]:
    """Parses a screenshot and returns icons for all reactions found."""
    icon_pages: Dict[int, List[numpy.ndarray]] = {}
    assertion_error: Optional[AssertionError] = None

    cap = cv2.VideoCapture(filename)
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break  # Video is over

            if not detect(frame):
                continue  # Skip frames not containing reactions.

            frame = to_720p(frame)
            try:
                new_icons = list(_parse_frame(frame))
                icon_pages[len(new_icons)] = new_icons
            except AssertionError as e:
                assertion_error = e
    finally:
        cap.release()

    if assertion_error and (is_image_file(filename) or not icon_pages):
        raise assertion_error

    return [icon for page in icon_pages.values() for icon in page]


def match_reactions(reaction_icons: List[numpy.ndarray]) -> Tuple[List[str], List[str]]:
    """Matches icons against database of reactions images, finding best matches."""
    matched_reactions = set()
    unmatched: List[str] = []
    if not reaction_icons:
        return [], unmatched

    reaction_db, templates, norms = _get_reaction_coarse_db()
    icon_vecs = numpy.stack([
        cv2.resize(ic, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        for ic in reaction_icons
    ]).astype(numpy.float32)

    for icon, icon_vec in zip(reaction_icons, icon_vecs):
        if len(reaction_db) > COARSE_TOP_K:
            d2 = norms - 2.0 * (templates @ icon_vec)
            top_idx = numpy.argpartition(d2, COARSE_TOP_K - 1)[:COARSE_TOP_K]
            candidates = [reaction_db[int(i)] for i in top_idx]
        else:
            candidates = reaction_db

        best_match, best_dist = _find_best_match(icon, candidates)
        if best_dist >= MAX_MATCH_DISTANCE:
            logging.warning(
                'Rejected reaction icon with L1 distance %.1f (closest: %s)',
                best_dist, best_match.reaction_name,
            )
            unmatched.append(f'{best_match.reaction_name}?d={best_dist:.1f}')
            continue
        matched_reactions.add(best_match.reaction_name)
    return sorted(matched_reactions), unmatched


def translate_names(reaction_names: List[str], locale: str) -> List[str]:
    """Translates a list of reaction names to the given locale."""
    if locale in ['auto', 'en-us']:
        return reaction_names

    translation_path = os.path.join('reactions', 'translations.json')
    with open(translation_path, encoding='utf-8') as fp:
        translations = json.load(fp)
    return [translations[name][locale] for name in reaction_names]


def _parse_frame(frame: numpy.ndarray) -> Iterator[numpy.ndarray]:
    """Extracts the individual reaction icons from the frame."""
    for x, y in REACTION_POSITIONS:
        # Skip empty slots.
        center_color = frame[y-6:y+6, x-6:x+6].mean(axis=(0, 1))
        if numpy.linalg.norm(center_color - EMPTY_COLOR) < 10:
            break
        if numpy.linalg.norm(center_color - SELECT_COLOR) < 20:
            break

        icon = frame[y-32:y+32, x-32:x+32]
        assert icon[34:42, 10:18].mean() < 250, 'Cursor is blocking a reaction.'
        assert icon[-5:, :, 2].mean() > 200, 'Tooltip is blocking a reaction.'

        # If the cursor is hovering on the icon, shrink it to normalize size.
        if icon[-3, -5, 1] > 227:
            icon = cv2.copyMakeBorder(
                icon, top=8, bottom=8, left=8, right=8,
                borderType=cv2.BORDER_CONSTANT, value=BG_COLOR)
            icon = cv2.resize(icon, (64, 64))

        yield icon


@functools.lru_cache()
def _get_reaction_db() -> List[ReactionImage]:
    """Fetches the reaction database for a given locale, with caching."""
    with open(os.path.join('reactions', 'names.json')) as fp:
        reaction_data = json.load(fp)
    return [ReactionImage(name, img) for name, img, _ in reaction_data]


@functools.lru_cache()
def _get_reaction_coarse_db() -> Tuple[List[ReactionImage], numpy.ndarray, numpy.ndarray]:
    """Precomputes 16x16 flattened float32 templates and squared norms for reactions."""
    reaction_db = _get_reaction_db()
    small = numpy.stack([
        cv2.resize(r.img, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        for r in reaction_db
    ]).astype(numpy.float32)
    norms = (small * small).sum(axis=1)
    return reaction_db, small, norms


def _find_best_match(icon: numpy.ndarray, candidates: List[ReactionImage]) -> Tuple[ReactionImage, float]:
    """Finds the closest matching reaction and its mean L1 pixel distance."""
    sims = [float(cv2.absdiff(icon, r.img).mean()) for r in candidates]
    if len(candidates) == 1 and sims[0] < MAX_MATCH_DISTANCE:
        return candidates[0], sims[0]

    if len(candidates) > 1:
        order = numpy.argsort(sims)
        if sims[int(order[1])] - sims[int(order[0])] > 3 and sims[int(order[0])] < MAX_MATCH_DISTANCE:
            best_idx = int(order[0])
            return candidates[best_idx], sims[best_idx]

    # Try small (x, y) shifts when the match is ambiguous or above threshold.
    shifted_icons = [
        numpy.roll(numpy.roll(icon, x, axis=1), y, axis=0)
        for x, y in itertools.product((-1, 0, 1), repeat=2)
    ]
    best_dists = [min(float(cv2.absdiff(si, r.img).mean()) for si in shifted_icons) for r in candidates]
    best_idx = int(numpy.argmin(best_dists))
    return candidates[best_idx], best_dists[best_idx]


if __name__ == "__main__":
    results = scan('examples/reactions.jpg')
    print('\n'.join(results.items))
