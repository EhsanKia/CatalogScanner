from common import ScanMode, ScanResult

import cv2
import itertools
import functools
import json
import numpy
import os

from typing import Dict, Iterator, List, Optional

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
    color = frame[370:380, 290:300].mean(axis=(0, 1))
    return numpy.linalg.norm(color - BG_COLOR) < 5


def scan(image_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans an image of reactions list and returns all reactions found."""
    reaction_icons = parse_image(image_file)
    reaction_names = match_reactions(reaction_icons)
    results = translate_names(reaction_names, locale)

    return ScanResult(
        mode=ScanMode.REACTIONS,
        items=results,
        locale=locale.replace('auto', 'en-us'),
    )


def parse_image(filename: str) -> List[ReactionImage]:
    """Parses a screenshot and returns icons for all reactions found."""
    icon_pages: Dict[int, List[ReactionImage]] = {}
    assertion_error: Optional[AssertionError] = None

    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        if frame.shape[:2] == (1080, 1920):
            frame = cv2.resize(frame, (1280, 720))

        if not detect(frame):
            continue  # Skip frames not containing reactions.

        try:
            new_icons = list(_parse_frame(frame))
            icon_pages[len(new_icons)] = new_icons
        except AssertionError as e:
            assertion_error = e

    if not icon_pages and assertion_error:
        raise assertion_error

    return itertools.chain.from_iterable(icon_pages.values())


def match_reactions(reaction_icons: List[ReactionImage]) -> List[str]:
    """Matches icons against database of reactions images, finding best matches."""
    matched_reactions = set()
    reaction_db = _get_reaction_db()
    for icon in reaction_icons:
        best_match = _find_best_match(icon, reaction_db)
        matched_reactions.add(best_match.reaction_name)
    return sorted(matched_reactions)


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


def _find_best_match(icon: numpy.ndarray, reactions: List[ReactionImage]) -> ReactionImage:
    """Finds the closest matching reaction for the given icon."""
    fast_similarity_metric = lambda r: cv2.absdiff(icon, r.img).mean()
    similarities = list(map(fast_similarity_metric, reactions))
    sim1, sim2 = numpy.partition(similarities, kth=2)[:2]

    # If the match seems obvious, return the quick result.
    if abs(sim1 - sim2) > 3:
        return reactions[numpy.argmin(similarities)]

    # Otherwise, we use a slower matching, which tries various shifts.
    def slow_similarity_metric(reaction):
        diffs = []
        for x, y in itertools.product([-1, 0, 1], repeat=2):
            shifted = numpy.roll(numpy.roll(icon, x, axis=1), y, axis=0)
            diffs.append(cv2.absdiff(shifted, reaction.img).sum())
        return min(diffs)  # Return lowest diff across shifts.

    similarities = list(map(slow_similarity_metric, reactions))
    return reactions[numpy.argmin(similarities)]


if __name__ == "__main__":
    results = scan('examples/reactions.jpg')
    print('\n'.join(results.items))
