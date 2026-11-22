from common import ScanMode, ScanResult, crop_720p_roi, slice_720p

import collections
import cv2
import functools
import json
import logging
import numpy
import os

from typing import Dict, Iterable, List, Optional, Tuple

# The expected color for the video background.
BG_COLOR = (194, 222, 228)
WOOD_COLOR = (115, 175, 228)
KITCHEN_COLOR = (160, 163, 243)

MAX_MATCH_DISTANCE = 35.0
COARSE_TOP_K = 16

# Items that look very alike and can be mixed up.
CONFUSED_ITEMS = {
    'concrete pillar': 'marble pillar',
    'marble pillar': 'concrete pillar',
}


class RecipeCard:
    """The image and data associated with a given recipe."""

    def __init__(self, item_name, filename, color_id):
        img_path = os.path.join('recipes', 'generated', filename)
        self.img = cv2.imread(img_path)
        self.name = item_name
        self.color_id = color_id

    def __repr__(self):
        return f'RecipeCard({self.name!r}, {self.color_id!r})'


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing DIY recipes."""
    color = slice_720p(frame, None, 20, 1200, 1240).mean(axis=(0, 1))
    if numpy.linalg.norm(color - WOOD_COLOR) < 15:
        raise AssertionError('Workbench scanning is not supported.')
    if numpy.linalg.norm(color - KITCHEN_COLOR) < 15:
        raise AssertionError('Kitchen scanning is not supported.')
    return bool(numpy.linalg.norm(color - BG_COLOR) < 15)


def scan(video_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans a video of scrolling through recipes list and returns all recipes found."""
    recipe_cards = parse_video(video_file)
    recipe_names, unmatched = match_recipes(recipe_cards)
    results = translate_names(recipe_names, locale)

    return ScanResult(
        mode=ScanMode.RECIPES,
        items=results,
        locale=locale.replace('auto', 'en-us'),
        unmatched=unmatched,
    )


def parse_video(filename: str) -> List[numpy.ndarray]:
    """Parses a whole video and returns images for all recipe cards found."""
    all_cards: List[numpy.ndarray] = []
    for frame in _read_frames(filename):
        for new_cards in _parse_frame(frame):
            if _is_duplicate_cards(all_cards, new_cards):
                continue  # Skip non-moving frames
            all_cards.extend(new_cards)
    return all_cards


def match_recipes(recipe_cards: List[numpy.ndarray]) -> Tuple[List[str], List[str]]:
    """Matches icons against database of recipe images, finding best matches."""
    matched_recipes = set()
    unmatched: List[str] = []
    valid_cards = [
        card for card in recipe_cards
        if numpy.linalg.norm(card[28:84, 28:84].mean(axis=(0, 1)) - BG_COLOR) >= 5
    ]
    if not valid_cards:
        return [], unmatched

    card_vecs = numpy.stack([
        cv2.resize(card, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        for card in valid_cards
    ]).astype(numpy.float32)

    for card, card_vec in zip(valid_cards, card_vecs):
        candidates = _get_candidate_recipes(card, card_vec)
        best_match, best_dist = _find_best_match(card, candidates)
        if best_dist >= MAX_MATCH_DISTANCE:
            logging.warning(
                'Rejected recipe card with L1 distance %.1f (closest: %s)',
                best_dist, best_match.name,
            )
            unmatched.append(f'{best_match.name}?d={best_dist:.1f}')
            continue

        item_name = best_match.name
        # If the item is already in our list, it might be confused with a similar item.
        if item_name in matched_recipes and item_name in CONFUSED_ITEMS:
            item_name = CONFUSED_ITEMS[item_name]

        matched_recipes.add(item_name)

    return sorted(matched_recipes), unmatched


def translate_names(recipe_names: List[str], locale: str) -> List[str]:
    """Translates a list of recipe names to the given locale."""
    if locale in ['auto', 'en-us']:
        return recipe_names

    translation_path = os.path.join('recipes', 'translations.json')
    with open(translation_path, encoding='utf-8') as fp:
        translations = json.load(fp)
    return [translations[name][locale] for name in recipe_names]


def _read_frames(filename: str, step: Optional[int] = None) -> Iterable[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region."""
    cap = cv2.VideoCapture(filename)
    if step is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, round(4 * fps / 30.0)) if fps > 0 else 4
    seen_valid = False
    can_skip = False
    pending_gap = 0
    total_gap_frames = 0
    try:
        while True:
            if can_skip and step > 1:
                for _ in range(step - 1):
                    if not cap.grab():
                        return

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            try:
                is_valid = detect(frame)
            except AssertionError:
                if not seen_valid:
                    raise
                is_valid = False

            if not is_valid:
                can_skip = False
                if seen_valid:
                    pending_gap += 1
                continue

            if pending_gap:
                total_gap_frames += pending_gap
                pending_gap = 0
            seen_valid = True
            can_skip = True
            yield crop_720p_roi(frame, 110, 720, 45, 730)
    finally:
        cap.release()
        if total_gap_frames:
            logging.warning('Skipped %d non-matching mid-video frames in %s', total_gap_frames, filename)


def _parse_frame(frame: numpy.ndarray) -> Iterable[List[numpy.ndarray]]:
    """Parses an individual frame and extracts cards from the recipe list."""
    # Start/end horizontal position for the 5 recipe cards.
    x_positions = [(11, 123), (148, 260), (286, 398), (423, 535), (560, 672)]

    # This code finds areas of the image that are beige (background color),
    # then it averages the frame across the Y-axis to find the area rows.
    # Lastly, it finds the y-positions marking the start/end of each row.
    thresh = cv2.inRange(frame, (185, 215, 218), (210, 230, 237))
    separators = numpy.nonzero(numpy.diff(thresh.mean(axis=1) > 195))[0]

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
    return {int(color_id): (b, g, r)
            for color_id, (r, g, b) in colors_data.items()}


@functools.lru_cache()
def _get_color_table() -> Tuple[List[int], numpy.ndarray]:
    """Returns color IDs and a float32 BGR matrix for vectorized distance lookup."""
    color_db = _get_color_db()
    color_ids = list(color_db.keys())
    color_arr = numpy.array([color_db[cid] for cid in color_ids], dtype=numpy.float32)
    return color_ids, color_arr


@functools.lru_cache()
def _get_recipe_coarse_db() -> Dict[int, Tuple[List[RecipeCard], numpy.ndarray, numpy.ndarray]]:
    """Precomputes 16x16 flattened float32 templates and squared norms per recipe color bucket."""
    coarse_db: Dict[int, Tuple[List[RecipeCard], numpy.ndarray, numpy.ndarray]] = {}
    for color_id, recipe_list in _get_recipe_db().items():
        small = numpy.stack([
            cv2.resize(r.img, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
            for r in recipe_list
        ]).astype(numpy.float32)
        norms = (small * small).sum(axis=1)
        coarse_db[color_id] = (recipe_list, small, norms)
    return coarse_db


def _get_candidate_recipes(card: numpy.ndarray, card_vec: numpy.ndarray) -> List[RecipeCard]:
    """Filters by card color and returns the top COARSE_TOP_K closest 16x16 candidates."""
    color_ids, color_arr = _get_color_table()
    coarse_db = _get_recipe_coarse_db()

    # Cut a small piece from the corner and calculate the average color.
    bg_color = card[104:107, 62:66, :].mean(axis=(0, 1))
    cdists = numpy.linalg.norm(color_arr - bg_color, axis=1)
    min_cd = cdists.min()
    matching_cids = [color_ids[i] for i in numpy.nonzero(cdists - min_cd <= 10)[0]]

    cand_scores: List[numpy.ndarray] = []
    cand_objs: List[RecipeCard] = []
    for cid in matching_cids:
        if cid not in coarse_db:
            continue
        bucket_recipes, templates, norms = coarse_db[cid]
        cand_scores.append(norms - 2.0 * (templates @ card_vec))
        cand_objs.extend(bucket_recipes)

    if len(cand_objs) <= COARSE_TOP_K:
        return cand_objs

    all_d2 = cand_scores[0] if len(cand_scores) == 1 else numpy.concatenate(cand_scores)
    top_indices = numpy.argpartition(all_d2, COARSE_TOP_K - 1)[:COARSE_TOP_K]
    return [cand_objs[int(i)] for i in top_indices]


def _find_best_match(card: numpy.ndarray, candidates: List[RecipeCard]) -> Tuple[RecipeCard, float]:
    """Finds the closest matching recipe and its mean L1 pixel distance."""
    sims = [float(cv2.absdiff(card, r.img).mean()) for r in candidates]
    if len(candidates) == 1 and sims[0] < MAX_MATCH_DISTANCE:
        return candidates[0], sims[0]

    if len(candidates) > 1:
        order = numpy.argsort(sims)
        if sims[int(order[1])] - sims[int(order[0])] > 3 and sims[int(order[0])] < MAX_MATCH_DISTANCE:
            best_idx = int(order[0])
            return candidates[best_idx], sims[best_idx]

    # Try small vertical shifts when the match is ambiguous or above threshold.
    shifted_cards = [numpy.roll(card, y, axis=0) for y in (-2, -1, 0, 1, 2)]
    best_dists = [min(float(cv2.absdiff(sc, r.img).mean()) for sc in shifted_cards) for r in candidates]
    best_idx = int(numpy.argmin(best_dists))
    return candidates[best_idx], best_dists[best_idx]


if __name__ == "__main__":
    results = scan('examples/recipes.mp4')
    print('\n'.join(results.items))
