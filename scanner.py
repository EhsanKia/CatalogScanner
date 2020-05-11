from absl import app
from absl import flags
from PIL import Image
from typing import Iterable, Iterator, List, Set

import cv2
import difflib
import json
import logging
import numpy
import pytesseract

LANG_MAP = {
    'zh-cn': 'chi_sim',
    'de-eu': 'deu',
    'en-eu': 'eng',
    'es-eu': 'spa',
    'fr-eu': 'fra',
    'it-eu': 'ita',
    'nl-eu': 'nld',
    'ru-eu': 'rus',
    'ja-jp': 'jpn',
    'ko-kr': 'kor',
    'zh-tw': 'chi_tra',
    'en-us': 'eng',
    'es-us': 'spa',
    'fr-us': 'fra',
}

flags.DEFINE_enum('lang', 'en-us', LANG_MAP.keys(), 'The language to use for parsing the item names.')


def read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region in grayscale."""
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Video is over

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (720, 1280), 'Invalid resolution: %s' % (gray.shape,)
        yield gray[150:630, 635:1050]  # The region containing the items
    cap.release()


def parse_frame(frame: numpy.ndarray) -> Iterator[numpy.ndarray]:
    """Parses an individual frame and extracts item rows from the list."""
    # Detect the dashed lines and iterate over pairs of dashed lines
    # Last line has dashes after but first line doesn't have dashes before,
    # therefore we prepend the list with zero for the starting line.
    lines = [0] + list((frame[:, 0] < 200).nonzero()[0])
    for y1, y2 in zip(lines, lines[1:]):
        if not (40 < y2 - y1 < 60):
            continue  # skip lines that are too close or far
        # Cut slightly below and above the dashed line
        yield frame[y2-40:y2-5, :]


def duplicate_rows(all_rows: List[numpy.ndarray], new_rows: List[numpy.ndarray]) -> bool:
    """Checks if the new set of rows are the same as the previous seen rows."""
    if not new_rows or len(all_rows) < len(new_rows):
        return False
    
    # Just check a middle row instead of all
    row_index = -len(new_rows) // 2
    diff = cv2.subtract(all_rows[row_index], new_rows[row_index])
    return diff.sum() < 100


def parse_video(filename: str) -> List[numpy.ndarray]:
    """Parses a whole video and returns all the item rows found."""
    all_rows: List[numpy.ndarray] = []
    for i, frame in enumerate(read_frames(filename)):
        if i % 3 != 0:
            continue  # Only parse every third frame
        new_rows = list(parse_frame(frame))
        if duplicate_rows(all_rows, new_rows):
            continue  # Skip non-moving frames
        all_rows.extend(new_rows)
    return all_rows


def run_tesseract(item_rows: List[numpy.ndarray], lang='eng') -> Set[str]:
    """Runs tesseract on the row images and returns list of unique items found."""
    assert item_rows, 'No items found, invalid video?'

    # Concatenate all rows and send a single image to Tesseract (OCR)
    concat_rows = cv2.vconcat(item_rows)

    # For larger catalogs, shrink size in half. Accuracy still remains as good.
    if concat_rows.shape[0] > 32000:
        concat_rows = cv2.resize(concat_rows, None, fx=0.5, fy=0.5)

    parsed_text = pytesseract.image_to_string(
        Image.fromarray(concat_rows), lang=lang)

    # Cleanup results a bit and try matching them again items using string distance
    return {t.strip().lower() for t in parsed_text.split('\n') if t.strip()}


def match_items(parsed_names: Iterable[str], item_db: Set[str]) -> Set[str]:
    """Matches a list of names against a database of items, finding best matches."""
    matched_items = set()
    no_match_count = 0
    for item in parsed_names:
        if item in item_db:
            # If item name exists is in the DB, add it as is
            matched_items.add(item)
            continue

        # Otherwise, try to find closest name in the DB witha cutoff
        matches = difflib.get_close_matches(item, item_db, n=1, cutoff=0.8)
        if not matches:
            logging.warning('No match found for %r', item)
            no_match_count += 1
            if no_match_count > 10:
                return set()
            continue
        logging.info('Matched %r to %r', item, matches[0])
        matched_items.add(matches[0])  # type: ignore
    return matched_items


def scan_catalog(video_file: str, lang_code: str = 'en-us') -> List[str]:
    """Scans a video of scrolling through a catalog and returns all items found."""
    item_rows = parse_video(video_file)
    item_names = run_tesseract(item_rows, lang=LANG_MAP[lang_code])

    with open('items/%s.json' % lang_code, encoding='utf-8') as fp:
        item_db = set(json.load(fp))
    clean_names = match_items(item_names, item_db)
    assert clean_names, 'Did not match most item names, wrong language?'

    return sorted(clean_names)


def main(argv):
    video_file = argv[1] if len(argv) > 1 else 'catalog.mp4'
    all_items = scan_catalog(video_file, lang_code=flags.FLAGS.lang)
    print('\n'.join(all_items))


if __name__ == "__main__":
    app.run(main)
