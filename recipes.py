from common import ScanMode, ScanResult

import collections
import cv2
import functools
import json
import numpy
import os

from typing import Dict, Iterable, List, Tuple

# The expected color for the video background.
BG_COLOR = (194, 222, 228)
WOOD_COLOR = (115, 175, 228)
KITCHEN_COLOR = (160, 167, 246)



class RecipeCard:
    """The image and data associated with a given recipe."""

    def __init__(self, item_name, filename, card_type):
        img_path = os.path.join('recipes', 'generated', filename)
        self.img = cv2.imread(img_path)
        self.item_name = item_name
        self.card_type = card_type

    def __repr__(self):
        return f'RecipeCard({self.item_name!r}, {self.card_type!r})'


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing DIY recipes."""
    color = frame[:20, 1200:1240].mean(axis=(0, 1))
    if numpy.linalg.norm(color - WOOD_COLOR) < 7:
        raise AssertionError('Workbench scanning is not supported.')
    if numpy.linalg.norm(color - KITCHEN_COLOR) < 7:
        raise AssertionError('Kitchen scanning is not supported.')
    return numpy.linalg.norm(color - BG_COLOR) < 7


def scan(video_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans a video of scrolling through recipes list and returns all recipes found."""
    recipe_cards = parse_video(video_file)
    recipe_names = match_recipes(recipe_cards)
    results = translate_names(recipe_names, locale)

    return ScanResult(
        mode=ScanMode.RECIPES,
        items=results,
        locale=locale.replace('auto', 'en-us'),
    )


def parse_video(filename: str) -> List[numpy.ndarray]:
    """Parses a whole video and returns images for all recipe cards found."""
    all_cards: List[numpy.ndarray] = []
    for i, frame in enumerate(_read_frames(filename)):
        if i % 4 != 0:
            continue  # Skip every 4th frame
        for new_cards in _parse_frame(frame):
            if _is_duplicate_cards(all_cards, new_cards):
                continue  # Skip non-moving frames
            all_cards.extend(new_cards)
    return all_cards


def match_recipes(recipe_cards: List[numpy.ndarray]) -> List[str]:
    """Matches icons against database of recipe images, finding best matches."""
    matched_recipes = set()
    for card in recipe_cards:
        # Check if the card is just the background color.
        card_center_color = card[28:84, 28:84].mean(axis=(0, 1))
        if numpy.linalg.norm(card_center_color - BG_COLOR) < 3:
            continue  # Skip blank card slots.

        possible_recipes = list(_get_candidate_recipes(card))
        best_match = _find_best_match(card, possible_recipes)
        matched_recipes.add(best_match.item_name)
    return sorted(matched_recipes)


def translate_names(recipe_names: List[str], locale: str) -> List[str]:
    """Translates a list of recipe names to the given locale."""
    if locale in ['auto', 'en-us']:
        return recipe_names

    translation_path = os.path.join('recipes', 'translations.json')
    with open(translation_path, encoding='utf-8') as fp:
        translations = json.load(fp)
    return [translations[name][locale] for name in recipe_names]


def _read_frames(filename: str) -> Iterable[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region."""
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        assert frame.shape[:2] == (720, 1280), \
            'Invalid resolution: {1}x{0}'.format(*frame.shape)

        if not detect(frame):
            continue  # Skip frames that are not showing recipes.

        # Crop the region containing recipe cards.
        yield frame[110:720, 45:730]
    cap.release()


def _parse_frame(frame: numpy.ndarray) -> Iterable[List[numpy.ndarray]]:
    """Parses an individual frame and extracts cards from the recipe list."""
    # Start/end horizontal position for the 5 recipe cards.
    x_positions = [(11, 123), (148, 260), (286, 398), (423, 535), (560, 672)]

    # This code finds areas of the image that are beige (background color),
    # then it averages the frame across the Y-axis to find the area rows.
    # Lastly, it finds the y-positions marking the start/end of each row.
    thresh = cv2.inRange(frame, (185, 215, 218), (210, 230, 237))
    separators = numpy.diff(thresh.mean(axis=1) > 195).nonzero()[0]

    # We do a first pass finding all sensible y positions.
    y_positions = []
    for y1, y2 in zip(separators, separators[1:]):
        if not (180 < y2 - y1 < 200):
            continue  # Invalid card size
        y_positions.append(y1)

    # Then, if the last row is missing, we predict its value.
    if y_positions and y_positions[-1] < 100:
        y_positions.append(y_positions[-1] + 211)

    for y1 in y_positions:
        row = []
        for x1, x2 in x_positions:
            card = frame[y1+37:y1+149, x1:x2]
            # Detects selected cards, which are bigger, and resizes them.
            if y1 > 9 and thresh[y1-10:y1-5, x1:x2].mean() < 100:
                card = frame[y1+22:y1+152, x1-9:x2+9]
                card = cv2.resize(card, (112, 112))
            row.append(card)
        yield row


def _is_duplicate_cards(all_cards: List[numpy.ndarray], new_cards: List[numpy.ndarray]) -> bool:
    """Checks if the new set of cards are the same as the previous seen cards."""
    if not new_cards or len(all_cards) < len(new_cards):
        return False

    new_concat = cv2.hconcat(new_cards)
    # Checks the last 3 rows for similarities to the newly added row.
    for ind in [slice(-5, None), slice(-10, -5), slice(-15, -10)]:
        old_concat = cv2.hconcat(all_cards[ind])
        if old_concat is None:
            continue
        if cv2.absdiff(new_concat, old_concat).mean() < 10:
            # Replace the old set with the new set.
            all_cards[ind] = new_cards
            return True
    return False


@functools.lru_cache()
def _get_recipe_db() -> Dict[int, List[RecipeCard]]:
    """Fetches the item database for a given locale, with caching."""
    with open(os.path.join('recipes', 'names.json')) as fp:
        recipes_data = json.load(fp)

    # Some recipes have alternate images, append those to the list.
    recipes_data.extend(
        (name, filename.replace('_0_0', '_1_0'), color)
        for name, filename, color in recipes_data if filename.endswith('_0_0.png'))

    recipe_db = collections.defaultdict(list)
    for item_name, filename, card_color in recipes_data:
        recipe = RecipeCard(item_name, filename, card_color)
        recipe_db[card_color].append(recipe)
    return recipe_db


@functools.lru_cache()
def _get_color_db() -> Dict[int, Tuple[int, int, int]]:
    """Fetches the item database for a given locale, with caching."""
    with open(os.path.join('recipes', 'colors.json')) as fp:
        colors_data = json.load(fp)
    return {int(color_id): tuple(reversed(rgb))
            for color_id, rgb in colors_data.items()}


def _get_candidate_recipes(card: numpy.ndarray) -> Iterable[RecipeCard]:
    """Guesses the recipe color and returns all recipes the card could be"""
    color_db = _get_color_db()
    recipe_db = _get_recipe_db()

    # Cut a small piece from the corner and calculate the average color.
    bg_color = card[104:107, 62:66, :].mean(axis=(0, 1))

    # Calculate how close each color is to the card's background color.
    distance_func = lambda x: numpy.linalg.norm(bg_color - color_db[x])
    color_distances = sorted((distance_func(c), c) for c in color_db)
    for distance, color_id in color_distances:
        # Stop if the candidate is much worse than best candidate.
        if distance - color_distances[0][0] > 10:
            break
            yield from recipe_db[color_id]


def _find_best_match(card: numpy.ndarray, recipes: List[RecipeCard]) -> RecipeCard:
    """Finds the closest matching recipe for the given card."""
    fast_similarity_metric = lambda r: cv2.absdiff(card, r.img).mean()
    similarities = list(map(fast_similarity_metric, recipes))
    sim1, sim2 = numpy.partition(similarities, kth=2)[:2]

    # If the match seems obvious, return the quick result.
    if abs(sim1 - sim2) > 3:
        return recipes[numpy.argmin(similarities)]

    # Otherwise, we use a slower matching, which tries various shifts.
    def slow_similarity_metric(recipe):
        diffs = []
        for y in [-1, 0, 1]:
            shifted = numpy.roll(card, y, axis=0)
            diffs.append(cv2.absdiff(shifted, recipe.img).sum())
        return min(diffs)  # Return lowest diff across shifts.

    similarities = list(map(slow_similarity_metric, recipes))
    return recipes[numpy.argmin(similarities)]


if __name__ == "__main__":
    results = scan('examples/recipes.mp4')
    print('\n'.join(results.items))
