from common import ScanMode, ScanResult

import cv2
import difflib
import functools
import json
import logging
import numpy
import pytesseract
import unicodedata

from PIL import Image
from typing import Dict, Iterator, List, Set

# The expected color for the video background.
BG_COLOR = numpy.array([110, 232, 238])

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


def scan_catalog(video_file: str, locale: str = 'en-us', for_sale: bool = False) -> ScanResult:
    """Scans a video of scrolling through a catalog and returns all items found."""
    item_rows = parse_video(video_file, for_sale)
    locale = _detect_locale(item_rows, locale)
    item_names = run_ocr(item_rows, lang=LOCALE_MAP[locale])
    results = match_items(item_names, locale)

    return ScanResult(
        mode=ScanMode.CATALOG,
        items=results,
        locale=locale,
    )


def parse_video(filename: str, for_sale: bool = False) -> numpy.ndarray:
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
        assert item_scroll_count < 20, 'Video is not page scrolling.'
        all_rows.extend(new_rows)

    assert all_rows, 'No items found, invalid video?'

    # Concatenate all rows into a single image.
    all_rows = _dedupe_rows(all_rows)
    return cv2.vconcat(all_rows)


def run_ocr(item_rows: numpy.ndarray, lang: str = 'eng') -> Set[str]:
    """Runs tesseract OCR on an image of item names and returns all items found."""
    # For larger catalogs, shrink size to avoid Tesseract's 32k limit.
    # Accuracy still remains good for most scripts.
    if item_rows.shape[0] > 32765:
        item_rows = cv2.resize(item_rows, None, fx=0.7, fy=0.7)

    parsed_text = pytesseract.image_to_string(
        Image.fromarray(item_rows), lang=lang, config=_get_tesseract_config(lang))

    # Split the results and remove empty lines.
    clean_items = {_cleanup_name(item, lang)
                   for item in parsed_text.split('\n')}
    return clean_items - {''}  # Remove empty lines


def match_items(item_names: Set[str], locale: str = 'en-us') -> List[str]:
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
            logging.warning('No matches found for %r', item)
            assert len(no_match_items) <= 25, \
                'Failed to match multiple items, wrong language?'
            continue

        # Calculate difference ratio for better logging
        ratio = difflib.SequenceMatcher(None, item, matches[0]).ratio()
        log_func = logging.info if ratio < 0.8 else logging.debug
        log_func('Matched %r to %r (%.2f)', item, matches[0], ratio)

        matched_items.add(matches[0])  # type: ignore

    return sorted(matched_items)


def _read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region in grayscale."""
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        assert frame.shape[:2] == (720, 1280), \
            'Invalid resolution: {1}x{0}'.format(*frame.shape)

        color = frame[:20, 1100:1150].mean(axis=(0, 1))
        if numpy.linalg.norm(color - BG_COLOR) > 5:
            continue  # Skip frames that are not showing items.

        # Turn to grayscale and crop the region containing item name and price.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield gray[150:630, 635:1220]
    cap.release()


def _parse_frame(frame: numpy.ndarray, for_sale: bool) -> Iterator[numpy.ndarray]:
    """Parses an individual frame and extracts item rows from the list."""
    # Detect the dashed lines and iterate over pairs of dashed lines
    # Last line has dashes after but first line doesn't have dashes before,
    # therefore we prepend the list with zero for the starting line.
    lines = [0] + list((frame[:, 0] < 200).nonzero()[0])
    for y1, y2 in zip(lines, lines[1:]):
        if not (40 < y2 - y1 < 60):
            continue  # skip lines that are too close or far

        # Cut row slightly below and above the dashed line
        row = frame[y2 - 40:y2 - 5, :]

        # Skip items that are not for sale (price region is lighter)
        if for_sale and row[:, 430:].min() > 100:
            continue

        yield row[:, :415]  # Return the name region


def _is_duplicate_rows(all_rows: List[numpy.ndarray], new_rows: List[numpy.ndarray]) -> bool:
    """Checks if the new set of rows are the same as the previous seen rows."""
    if not new_rows or len(all_rows) < len(new_rows):
        return False

    row_index = -len(new_rows) // 2  # Just check a middle row instead of all
    diff = cv2.absdiff(all_rows[row_index], new_rows[row_index])
    return diff.mean() < 2


def _is_item_scroll(all_rows: List[numpy.ndarray], new_rows: List[numpy.ndarray]) -> bool:
    """Checks whether the video is item scrolling instead of page scrolling."""
    if len(all_rows) < 3 or len(new_rows) < 3:
        return False

    # Items move by only one position when item scrolling.
    diff = cv2.absdiff(all_rows[-2], new_rows[-3])
    return diff.mean() < 5


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


def _detect_locale(item_rows: numpy.ndarray, locale: str) -> str:
    """Detects the right locale for the given items if required."""
    if locale != 'auto':
        # If locale is already specified, return as is.
        return locale

    # Convert to Pillow image and truncate overly long images.
    image = Image.fromarray(item_rows[:9800, :])

    try:
        osd_data = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return 'en-us'

    possible_locales = SCRIPT_MAP.get(osd_data['script'])
    assert possible_locales, 'Failed to automatically detect language.'

    # If we can uniquely guess the language from the script, use that.
    if len(possible_locales) == 1:
        logging.info('Detected locale: %s', possible_locales[0])
        return possible_locales[0]

    # Otherwise, run OCR on the first few items and try to find the best matching locale.
    item_names = run_ocr(item_rows[:30 * 35, :], lang='script/Latin')

    def match_score_func(locale):
        """Computes how many items match for a given locale."""
        item_db = _get_item_db(locale)
        return sum(name in item_db for name in item_names)

    best_locale = max(possible_locales, key=match_score_func)
    logging.info('Detected locale: %s', best_locale)
    return best_locale


if __name__ == "__main__":
    results = scan_catalog('videos/catalog.mp4')
    print('\n'.join(results.items))
