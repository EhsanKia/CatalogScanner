from common import ScanMode, ScanResult

import cv2
import difflib
import functools
import json
import logging
import numpy
import pytesseract
import random
import typing
import unicodedata

from PIL import Image
from typing import Dict, Iterator, List, Set, Tuple

# The expected color for the video background.
TOP_COLOR = (110, 233, 238)
SIDE_COLOR = (180, 253, 254)

# Bad background colors
WARDELL_COLOR = (211, 214, 248)
NOOK_MILES_COLOR = (243, 207, 200)

# Mapping supported AC:NH locales to tesseract languages.
LOCALE_MAP: Dict[str, str] = {
    'auto': 'auto',  # Automatic detection
    'de-eu': 'deu',
    'en-eu': 'eng',
    'en-us': 'eng',
    'es-eu': 'spa',
    'es-us': 'spa',
    'fr-eu': 'fra',
    'fr-us': 'fra',
    'it-eu': 'ita',
    'ja-jp': 'jpn',
    'ko-kr': 'kor',
    'nl-eu': 'nld',
    'ru-eu': 'rus',
    'zh-cn': 'chi_sim',
    'zh-tw': 'chi_tra',
}

# Mapping of Tesseract scripts to possible locales.
SCRIPT_MAP: Dict[str, List[str]] = {
    'Japanese': ['ja-jp'],
    'Cyrillic': ['ru-eu'],
    'HanS': ['zh-cn'],
    'HanT': ['zh-tw'],
    'Hangul': ['ko-kr'],
    'Latin': ['en-us', 'en-eu', 'fr-eu', 'fr-us', 'de-eu',
              'es-eu', 'es-us', 'it-eu', 'nl-eu']
}


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing Nook Shopping catalog."""
    side_color = frame[150:160, -20:].mean(axis=(0, 1))
    if numpy.linalg.norm(side_color - WARDELL_COLOR) < 10:
        raise AssertionError('Wardell catalog is not supported.')
    if numpy.linalg.norm(side_color - NOOK_MILES_COLOR) < 10:
        raise AssertionError('Nook Miles catalog is not supported.')
    return numpy.linalg.norm(side_color - SIDE_COLOR) < 10


def scan(video_file: str, locale: str = 'en-us', for_sale: bool = False) -> ScanResult:
    """Scans a video of scrolling through a catalog and returns all items found."""
    item_rows = parse_video(video_file, for_sale)
    locale = _detect_locale(item_rows, locale)
    item_names = run_ocr(item_rows, lang=LOCALE_MAP[locale])
    results, unmatched = match_items(item_names, locale)

    return ScanResult(
        mode=ScanMode.CATALOG,
        items=results,
        locale=locale,
        unmatched=unmatched,
    )


def parse_video(filename: str, for_sale: bool = False) -> List[numpy.ndarray]:
    """Parses a whole video and returns an image containing all the items found."""
    unfinished_page = False
    item_scroll_count = 0
    all_rows: List[numpy.ndarray] = []
    for i, frame in enumerate(_read_frames(filename)):
        if not unfinished_page and i % 3 != 0:
            continue  # Only parse every third frame (3 frames per page)
        new_rows = list(_parse_frame(frame, for_sale))
        if _is_duplicate_rows(all_rows, new_rows):
            continue  # Skip non-moving frames

        # There's an issue in Switch's font rendering where it struggles to
        # keep up with page scrolling, leading to bottom rows sometimes being empty.
        # Since we parse every third frame, this can lead to items getting missed.
        # The fix is to search for empty rows and force a scan of the next frame.
        unfinished_page = any(r.min() > 150 for r in new_rows)

        # Exit if video is not properly page scrolling.
        item_scroll_count += _is_item_scroll(all_rows, new_rows)
        assert item_scroll_count < 20, 'Video is scrolling too slowly.'
        all_rows.extend(new_rows)

    assert all_rows, 'No items found, invalid video?'

    # Concatenate all rows into a single image.
    return _dedupe_rows(all_rows)


def run_ocr(item_rows: List[numpy.ndarray], lang: str = 'eng') -> Set[str]:
    """Runs tesseract OCR on an image of item names and returns all items found."""
    if not item_rows:
        return set()  # Recursive base case.

    # For larger catalogs, recursively split scans to avoid Tesseract's 32k limit.
    # Each row is 35px high; 900 x 35 = 31.5k which is below the limit.
    item_rows, remaining_rows = item_rows[:900], item_rows[900:]

    logging.debug('Running Tesseract on %s rows', len(item_rows))
    parsed_text = pytesseract.image_to_string(
        Image.fromarray(cv2.vconcat(item_rows)),
        lang=lang, config=_get_tesseract_config(lang))
    assert isinstance(parsed_text, str), 'Tesseract returned bytes'

    # Split the results and remove empty lines.
    clean_names = {_cleanup_name(item, lang)
                   for item in parsed_text.split('\n')}

    # Add recursive results and remove empty lines.
    remaining_names = run_ocr(remaining_rows, lang)
    return (clean_names | remaining_names) - {''}


def match_items(item_names: Set[str], locale: str = 'en-us') -> Tuple[List[str], List[str]]:
    """Matches a list of names against a database of items, finding best matches."""
    no_match_items = []
    matched_items = set()
    item_db = _get_item_db(locale)
    for item in sorted(item_names):
        if item in item_db:
            # If item name exists is in the DB, add it as is
            matched_items.add(item)
            continue

        # Otherwise, try to find closest name in the DB with a cutoff.
        matches = difflib.get_close_matches(item, item_db, n=1, cutoff=0.5)
        if not matches:
            no_match_items.append(item)
            assert len(no_match_items) <= 0.3 * len(item_names), \
                'Failed to match multiple items, wrong language?'
            continue

        # Calculate difference ratio for better logging
        ratio = difflib.SequenceMatcher(None, item, matches[0]).ratio()
        logging.debug('Matched %r to %r (%.2f)', item, matches[0], ratio)

        matched_items.add(matches[0])  # type: ignore

    if no_match_items:
        logging.warning('Failed to match %d items: %s', len(no_match_items), no_match_items)

    return sorted(matched_items), no_match_items


def _read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region in grayscale."""
    scroll_positions = []
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        assert frame.shape[:2] == (720, 1280), \
            'Invalid resolution: {1}x{0}'.format(*frame.shape)

        if not detect(frame):
            scroll_positions = []  # Reset scroll positions on catalog change.
            continue  # Skip frames where item list is not visible.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop scrollbar region and get scroll position, then warn about bad scrolling.
        scrollbar = gray[160:570, 1235:1245].mean(axis=1)
        scroll_positions.append(numpy.argmax(scrollbar < 150))
        if _is_inconsistent_scroll(scroll_positions):
            raise AssertionError('Video is scrolling inconsistently.')

        # Crop the region containing item name and price.
        yield gray[150:630, 635:1220]
    cap.release()


def _parse_frame(frame: numpy.ndarray, for_sale: bool) -> Iterator[numpy.ndarray]:
    """Parses an individual frame and extracts item rows from the list."""
    # Detect the dashed lines and iterate over pairs of dashed lines
    # Last line has dashes after but first line doesn't have dashes before,
    # therefore we prepend the list with zero for the starting line.
    y_lines = list((frame[:, 0] < 200).nonzero()[0])
    if not y_lines:
        return

    # Normalize row lines by taking the average of all of them.
    # We know they are 53.45px apart, so we find the best offset from given lines.
    centers = [numpy.fmod(y, 53.45) for y in y_lines]
    centroid = round(numpy.median(centers))
    y_positions = numpy.arange(centroid, 480, 53.45).astype(int)

    for y in y_positions:
        if y < 40:
            # Skip rows that are offscreen
            continue

        # Cut row slightly below and above the dashed line
        row = frame[y - 40:y - 5, :]

        # Skip items that are not for sale (price region is lighter)
        if for_sale and row[:, 430:].min() > 100:
            continue

        yield row[:, :415]  # Return the name region


def _is_duplicate_rows(all_rows: List[numpy.ndarray], new_rows: List[numpy.ndarray]) -> bool:
    """Checks if the new set of rows are the same as the previous seen rows."""
    if not len(all_rows) > len(new_rows) > 4:
        return False

    # Check a few middle rows to avoid the hovered row.
    old_concat = cv2.vconcat(all_rows[-5:-2])
    new_concat = cv2.vconcat(new_rows[-5:-2])
    diff = cv2.absdiff(old_concat, new_concat)
    return diff.mean() < 4


def _is_item_scroll(all_rows: List[numpy.ndarray], new_rows: List[numpy.ndarray]) -> bool:
    """Checks whether the video is item scrolling instead of page scrolling."""
    if len(all_rows) < 3 or len(new_rows) < 3:
        return False

    # Items move by only one position when item scrolling.
    if cv2.absdiff(all_rows[-2], new_rows[-3]).mean() < 4:
        return True
    if cv2.absdiff(all_rows[-3], new_rows[-2]).mean() < 4:
        return True
    return False


def _is_inconsistent_scroll(scroll_positions: List[int]) -> bool:
    """Detect when the user is not scrolling in a consistent direction."""
    scroll_deltas = typing.cast(numpy.ndarray, numpy.diff(scroll_positions))
    downscroll_count = numpy.count_nonzero(scroll_deltas > 0)
    upscroll_count = numpy.count_nonzero(scroll_deltas < 0)
    return downscroll_count > 10 and upscroll_count > 10


def _dedupe_rows(all_rows: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """Dedupe rows by using image hashing and remove blank rows."""
    row_set: Set[str] = set()
    deduped_rows: List[numpy.ndarray] = []
    for row in all_rows:
        if row.min() > 150:
            continue  # Blank row
        row_hash = str(cv2.img_hash.blockMeanHash(row, mode=1)[0])
        if row_hash in row_set:
            continue  # Row already seen
        row_set.add(row_hash)
        deduped_rows.append(row)
    return deduped_rows


def _get_tesseract_config(lang: str) -> str:
    """Generates Tesseract configurations for the given language."""
    configs = [
        '--psm 6'  # Manually specify that we know orientation / shape.
        '-c preserve_interword_spaces=1',  # Fixes spacing between logograms.
        '-c tessedit_do_invert=0',  # Speed up skipping invert check.
    ]
    if lang in ['jpn', 'chi_sim', 'chi_tra']:
        # Parameters specific to parsing logograms.
        configs.extend([
            '-c language_model_ngram_on=0',
            '-c textord_force_make_prop_words=F',
            '-c edges_max_children_per_outline=40',
        ])
    return ' '.join(configs)


def _cleanup_name(item_name: str, lang: str) -> str:
    """Applies some manual name cleanup to fix OCR issues and improve matching."""
    item_name = item_name.strip()
    item_name = item_name.replace('Ao dai', 'Áo dài')
    item_name = item_name.replace('Bail', 'Ball')

    # Normalize unicode characters for better matching.
    item_name = unicodedata.normalize('NFKC', item_name)

    if lang == 'rus':
        # Fix Russian matching of Nook Inc.
        item_name = item_name.replace('Моок', 'Nook')
        item_name = item_name.replace('пс.', 'Inc.')
        item_name = item_name.replace('тс.', 'Inc.')

    return item_name


@functools.lru_cache(maxsize=None)
def _get_item_db(locale: str) -> Set[str]:
    """Fetches the item database for a given locale, with caching."""
    with open('items/%s.json' % locale, encoding='utf-8') as fp:
        return set(json.load(fp))


def _detect_locale(item_rows: List[numpy.ndarray], locale: str) -> str:
    """Detects the right locale for the given items if required."""
    if locale != 'auto':
        # If locale is already specified, return as is.
        return locale

    # Sample a subset of the rows and convert to Pillow image.
    if len(item_rows) > 300:
        item_rows = random.sample(item_rows, 300)
    image = Image.fromarray(cv2.vconcat(item_rows))

    try:
        osd_data = typing.cast(Dict[str, str], pytesseract.image_to_osd(
            image, output_type=pytesseract.Output.DICT))
    except pytesseract.TesseractError:
        return 'en-us'

    possible_locales = SCRIPT_MAP.get(osd_data['script'])
    assert possible_locales, 'Failed to automatically detect language.'

    # If we can uniquely guess the language from the script, use that.
    if len(possible_locales) == 1:
        logging.info('Detected locale: %s', possible_locales[0])
        return possible_locales[0]

    # Otherwise, run OCR on the first few items and try to find the best matching locale.
    if len(item_rows) > 30:
        item_rows = random.sample(item_rows, 30)
    item_names = run_ocr(item_rows, lang='script/Latin')

    def match_score_func(locale):
        """Computes how many items match for a given locale."""
        item_db = _get_item_db(locale)
        return sum(name in item_db for name in item_names)

    best_locale = max(possible_locales, key=match_score_func)
    logging.info('Detected locale: %s', best_locale)
    return best_locale


if __name__ == "__main__":
    results = scan('examples/catalog.mp4')
    print('\n'.join(results.items))
