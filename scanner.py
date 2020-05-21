from absl import app
from absl import flags
from PIL import Image
from typing import Iterable, Iterator, List, Optional, Set

import cv2
import difflib
import json
import logging
import numpy
import pytesseract

LANG_MAP = {
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

flags.DEFINE_enum('lang', 'en-us', LANG_MAP.keys(), 'The language to use for parsing the item names.')
flags.DEFINE_bool('for_sale', False, 'If true, the scanner will filter items that are not for sale.')


def read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region in grayscale."""
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (720, 1280), 'Invalid resolution: %s' % (gray.shape,)
        yield gray[150:630, 635:1220]  # The region containing the item name and price
    cap.release()


def parse_frame(frame: numpy.ndarray, for_sale: bool = False) -> Iterator[numpy.ndarray]:
    """Parses an individual frame and extracts item rows from the list."""
    # Detect the dashed lines and iterate over pairs of dashed lines
    # Last line has dashes after but first line doesn't have dashes before,
    # therefore we prepend the list with zero for the starting line.
    lines = [0] + list((frame[:, 0] < 200).nonzero()[0])
    for y1, y2 in zip(lines, lines[1:]):
        if not (40 < y2 - y1 < 60):
            continue  # skip lines that are too close or far

        # Cut row slightly below and above the dashed line
        row = frame[y2-40:y2-5, :]

        # Skip items that are not for sale (price region is lighter)
        if for_sale and row[:, 430:].min() > 100:
            continue

        yield row[:, :415]  # Return the name region


def duplicate_rows(all_rows: List[numpy.ndarray], new_rows: List[numpy.ndarray]) -> bool:
    """Checks if the new set of rows are the same as the previous seen rows."""
    if not new_rows or len(all_rows) < len(new_rows):
        return False

    # Just check a middle row instead of all
    row_index = -len(new_rows) // 2
    diff = cv2.subtract(all_rows[row_index], new_rows[row_index])
    return diff.mean() < 2


def parse_video(filename: str, for_sale: bool = False) -> List[numpy.ndarray]:
    """Parses a whole video and returns all the item rows found."""
    unfinished_page = False
    all_rows: List[numpy.ndarray] = []
    for i, frame in enumerate(read_frames(filename)):
        if not unfinished_page and i % 3 != 0:
            continue  # Only parse every third frame (3 frames per page)
        new_rows = list(parse_frame(frame, for_sale))
        if duplicate_rows(all_rows, new_rows):
            continue  # Skip non-moving frames

        # There's an issue in Switch's font rendering where it struggles to
        # keep up with page scrolling, leading to bottom rows sometimes being empty.
        # Since we parse every third frame, this can lead to items getting missed.
        # The fix is to search for empty rows and force a scan of the next frame.
        unfinished_page = any(r.min() > 150 for r in new_rows)

        all_rows.extend(new_rows)
    return all_rows


def get_tesseract_config(lang: str) -> str:
    configs = [
        '--psm 6'  # Manually specify that we know orientation / shape.
        '-c preserve_interword_spaces=1',  # Fixes spacing between logograms.
        '-c tessedit_do_invert=0',  # Speed up skipping invert check.
    ]
    if LANG_MAP.get(lang, lang) in ['jpn', 'chi_sim', 'chi_tra']:
        configs.extend([
            '-c language_model_ngram_on=0',
            '-c textord_force_make_prop_words=F',
            '-c edges_max_children_per_outline=40',
        ])
    return ' '.join(configs)

def run_tesseract(item_rows: List[numpy.ndarray], lang: str = 'eng') -> Set[str]:
    """Runs tesseract on the row images and returns list of unique items found."""
    assert item_rows, 'No items found, invalid video?'

    # Concatenate all rows to send a single image to Tesseract
    concat_rows = cv2.vconcat(item_rows)

    # For larger catalogs, shrink size in half. Accuracy still remains as good.
    if concat_rows.shape[0] > 32000:
        concat_rows = cv2.resize(concat_rows, None, fx=0.5, fy=0.5)

    parsed_text = pytesseract.image_to_string(
        Image.fromarray(concat_rows), lang=lang, config=get_tesseract_config(lang))

    # Split the results and remove empty lines.
    return set(map(str.strip, parsed_text.split('\n'))) - {''}


def match_items(parsed_names: Set[str], item_db: Set[str]) -> Set[str]:
    """Matches a list of names against a database of items, finding best matches."""
    no_match_count = 0
    matched_items = set()
    for item in parsed_names:
        if item in item_db:
            # If item name exists is in the DB, add it as is
            matched_items.add(item)
            continue

        # Otherwise, try to find closest name in the DB witha cutoff
        matches = difflib.get_close_matches(item, item_db, n=1, cutoff=0.6)
        if not matches:
            logging.warning('No match found for %r', item)
            no_match_count += 1
            assert no_match_count <= 20, \
                'Failed to match multiple items, wrong language?'
            continue
        logging.info('Matched %r to %r', item, matches[0])
        matched_items.add(matches[0])  # type: ignore

    if no_match_count:
        logging.warning('%d items failed to match.', no_match_count)
    return matched_items


def scan_catalog(video_file: str, lang_code: str = 'en-us', for_sale: bool = False) -> List[str]:
    """Scans a video of scrolling through a catalog and returns all items found."""
    item_rows = parse_video(video_file, for_sale)
    item_names = run_tesseract(item_rows, lang=LANG_MAP[lang_code])

    with open('items/%s.json' % lang_code, encoding='utf-8') as fp:
        item_db = set(json.load(fp))
    clean_names = match_items(item_names, item_db)
    return sorted(clean_names)


def main(argv):
    video_file = argv[1] if len(argv) > 1 else 'catalog.mp4'
    all_items = scan_catalog(
        video_file,
        lang_code=flags.FLAGS.lang,
        for_sale=flags.FLAGS.for_sale,
    )
    print('\n'.join(all_items))


if __name__ == "__main__":
    app.run(main)
