from common import ScanMode, ScanResult, iter_frames

import concurrent.futures
import contextlib
import cv2
import functools
import json
import logging
import numpy
import os
import pytesseract
import re
import typing
import unicodedata

from PIL import Image
from rapidfuzz import fuzz, process
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

# Prevent OpenMP-enabled Tesseract builds from oversubscribing 1-2 vCPU hosts.
os.environ.setdefault('OMP_THREAD_LIMIT', '1')

# Maximum rows per Tesseract invocation and parallel worker cap.
OCR_CHUNK_SIZE = 250
MAX_OCR_WORKERS = min(4, os.cpu_count() or 2)

# The expected color for the video background (BGR).
TOP_COLOR = (110, 233, 238)
SIDE_COLOR = (180, 253, 254)

# Bad background colors (BGR).
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

# Candidate OCR models and locales used for deterministic DB-scored locale detection.
DETECTION_CANDIDATES: Dict[str, List[str]] = {
    'script/Latin': [
        'en-us', 'en-eu', 'es-eu', 'es-us',
        'fr-eu', 'fr-us', 'de-eu', 'it-eu', 'nl-eu',
    ],
    'jpn': ['ja-jp'],
    'chi_sim': ['zh-cn'],
    'chi_tra': ['zh-tw'],
    'kor': ['ko-kr'],
    'rus': ['ru-eu'],
}

LOCALE_TIEBREAK_ORDER: List[str] = [
    'en-us', 'en-eu', 'zh-tw', 'zh-cn', 'ja-jp', 'ko-kr',
    'fr-eu', 'fr-us', 'de-eu', 'es-eu', 'es-us', 'it-eu', 'nl-eu', 'ru-eu',
]

# Items renamed across AC:NH patches (e.g. 1.3 / 2.0) mapped to current canonical names.
LEGACY_ALIASES: Dict[str, Dict[str, str]] = {
    'en-us': {
        'Writing desk': 'Study desk',
        'Writing chair': 'Study chair',
        'Writing poster': 'Study poster',
        'Teacup ride': 'Plaza teacup ride',
        'Fish and chips': 'Fish-and-chips',
        'Tee with silicon bib': 'Tee with silicone bib',
    },
    'en-eu': {
        'Writing desk': 'Study desk',
        'Writing chair': 'Study chair',
        'Writing poster': 'Study poster',
        'Teacup ride': 'Plaza teacup ride',
        'Fish and chips': 'Fish-and-chips',
        'Tee with silicon bib': 'Tee with silicone bib',
    },
}

# Player-specific tent/moving-kit names contain arbitrary player names and should be ignored.
PLAYER_ITEM_PATTERNS: Dict[str, re.Pattern] = {
    'en-us': re.compile(r"^.+'s (?:tent|kit)$", re.IGNORECASE),
    'en-eu': re.compile(r"^.+'s (?:tent|kit)$", re.IGNORECASE),
    'de-eu': re.compile(r'^(?:Zelt|Kit) von .+$', re.IGNORECASE),
    'es-eu': re.compile(r'^(?:Tienda de|Kit de) [A-ZÁÉÍÓÚÑ].+$'),
    'es-us': re.compile(r'^(?:Tienda|Kit de) [A-ZÁÉÍÓÚÑ].+$'),
    'fr-eu': re.compile(r'^(?:Tente de .+|Kit de [A-ZÀ-Ÿ].+)$'),
    'fr-us': re.compile(r'^(?:Tente de .+|Kit de [A-ZÀ-Ÿ].+)$'),
    'it-eu': re.compile(r'^(?:Tenda di .+|Kit di [A-ZÀ-Ÿ].+)$'),
    'nl-eu': re.compile(r'^Tent van .+$', re.IGNORECASE),
    'ru-eu': re.compile(r'^(?:Палатка .+|Набор [А-ЯЁA-Z].+)$'),
    'ja-jp': re.compile(r'^.+のテント$'),
    'ko-kr': re.compile(r'^.+(?:의 텐트| 이전 키트)$'),
    'zh-cn': re.compile(r'^.+的帐篷$'),
    'zh-tw': re.compile(r'^.+的帳篷$'),
}

_BAIL_RE = re.compile(r'\b([Bb])ail\b')


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing Nook Shopping catalog."""
    side_color = frame[150:160, -20:].mean(axis=(0, 1))
    if numpy.linalg.norm(side_color - WARDELL_COLOR) < 20:
        raise AssertionError('Wardell catalog is not supported.')
    if numpy.linalg.norm(side_color - NOOK_MILES_COLOR) < 20:
        raise AssertionError('Nook Miles catalog is not supported.')
    if numpy.linalg.norm(side_color - SIDE_COLOR) >= 22:
        return False
    top_color = frame[:20, -20:].mean(axis=(0, 1))
    return bool(numpy.linalg.norm(top_color - TOP_COLOR) < 45)


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
    """Parses a whole video and returns deduplicated row images for all items found."""
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Native 30fps Switch captures stay on each page for ~3 frames; lower FPS needs smaller step.
    frame_step = 3 if (fps <= 0 or fps >= 29.0) else (2 if fps >= 20.0 else 1)

    unfinished_page = False
    item_scroll_count = 0
    all_rows: List[numpy.ndarray] = []
    with contextlib.closing(_read_frames(cap, log_label=filename)) as frames:
        for i, frame in enumerate(frames):
            if not unfinished_page and i % frame_step != 0:
                continue  # Skip intermediate frames on the same page
            new_rows = list(_parse_frame(frame, for_sale))
            if _is_duplicate_rows(all_rows, new_rows):
                continue  # Skip non-moving frames

            # Switch's font rendering sometimes lags fast page scrolling, leaving bottom rows blank.
            unfinished_page = any(r.min() > 150 for r in new_rows)

            # Exit if video is not properly page scrolling.
            item_scroll_count += _is_item_scroll(all_rows, new_rows)
            assert item_scroll_count < 20, 'Video is scrolling too slowly.'
            all_rows.extend(new_rows)

    assert all_rows, 'No items found, invalid video?'

    return _dedupe_rows(all_rows)


def _ocr_chunk(chunk: List[numpy.ndarray], lang: str, config: str) -> Set[str]:
    """Runs Tesseract OCR on a single chunk of vertically concatenated row strips."""
    parsed_text = pytesseract.image_to_string(
        Image.fromarray(cv2.vconcat(chunk)),
        lang=lang,
        config=config,
    )
    assert isinstance(parsed_text, str), 'Tesseract returned bytes'
    return {_cleanup_name(item, lang) for item in parsed_text.split('\n')} - {''}


def run_ocr(item_rows: List[numpy.ndarray], lang: str = 'eng') -> Set[str]:
    """Runs Tesseract OCR across item rows in parallel chunks and returns all parsed lines."""
    if not item_rows:
        return set()

    chunks = [
        item_rows[i:i + OCR_CHUNK_SIZE]
        for i in range(0, len(item_rows), OCR_CHUNK_SIZE)
    ]
    config = _get_tesseract_config(lang)
    logging.debug('Running Tesseract (%s) on %d rows across %d chunk(s)', lang, len(item_rows), len(chunks))

    if len(chunks) == 1 or MAX_OCR_WORKERS <= 1:
        return _ocr_chunk(chunks[0], lang, config) if len(chunks) == 1 else set().union(
            *(_ocr_chunk(c, lang, config) for c in chunks)
        )

    workers = min(len(chunks), MAX_OCR_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_ocr_chunk, chunk, lang, config) for chunk in chunks]
        results: Set[str] = set()
        for future in futures:
            results |= future.result()
        return results


def _fuzzy_thresholds(length: int) -> Tuple[float, float, float]:
    """Returns (min_score_cutoff, min_margin_over_second, high_confidence_bypass) on 0-100 scale."""
    if length <= 2:
        return 100.0, 100.0, 100.0
    if length == 3:
        return 66.0, 8.0, 90.0
    if length == 4:
        return 70.0, 5.0, 88.0
    if length <= 6:
        return 65.0, 4.0, 86.0
    return 56.0, 3.5, 88.0


def match_items(item_names: Set[str], locale: str = 'en-us') -> Tuple[List[str], List[str]]:
    """Matches parsed OCR strings against the locale item database using exact + guarded fuzzy matching."""
    no_match_items: List[str] = []
    matched_items: Set[str] = set()
    item_db = _get_item_db(locale)
    item_list = _get_item_list(locale)
    aliases = LEGACY_ALIASES.get(locale, {})
    player_pattern = PLAYER_ITEM_PATTERNS.get(locale)

    for raw_item in sorted(item_names):
        item = aliases.get(raw_item, raw_item)
        if item in item_db:
            matched_items.add(item)
            continue

        if player_pattern and player_pattern.match(item):
            logging.debug('Skipping player-named item %r', raw_item)
            continue

        cutoff, min_margin, high_conf = _fuzzy_thresholds(len(item))
        if cutoff >= 100.0:
            no_match_items.append(raw_item)
            assert len(no_match_items) <= max(3, int(0.3 * len(item_names))), \
                'Failed to match multiple items, wrong language?'
            continue

        candidates = process.extract(item, item_list, scorer=fuzz.ratio, limit=2, score_cutoff=cutoff)
        if candidates:
            best_name, best_score, _ = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0
            if best_score >= high_conf or (best_score - second_score) >= min_margin:
                resolved = aliases.get(best_name, best_name)
                logging.debug('Matched %r to %r (%.1f vs %.1f)', raw_item, resolved, best_score, second_score)
                matched_items.add(resolved)
                continue

        no_match_items.append(raw_item)
        assert len(no_match_items) <= max(3, int(0.3 * len(item_names))), \
            'Failed to match multiple items, wrong language?'

    if no_match_items:
        logging.warning('Failed to match %d items: %s', len(no_match_items), no_match_items)

    return sorted(matched_items), no_match_items


def _read_frames(
    source: Union[str, cv2.VideoCapture],
    log_label: Optional[str] = None,
) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the relevant region in grayscale."""
    scroll_positions: List[int] = []
    for frame, gap_before in iter_frames(source, detect, log_label=log_label):
        if gap_before >= 5:
            scroll_positions = []  # Reset scroll positions on catalog section change.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop scrollbar region and get scroll position, then warn about bad scrolling.
        scrollbar = gray[160:570, 1235:1245].mean(axis=1)
        dark = scrollbar < 150
        if dark.any():
            scroll_positions.append(int(numpy.argmax(dark)))
            if _is_inconsistent_scroll(scroll_positions):
                raise AssertionError('Video is scrolling inconsistently.')

        # Crop the region containing item name and price.
        yield gray[150:630, 635:1220]


def _parse_frame(frame: numpy.ndarray, for_sale: bool) -> Iterator[numpy.ndarray]:
    """Parses an individual frame and extracts item rows from the list."""
    y_lines = list((frame[:, 0] < 200).nonzero()[0])
    if not y_lines:
        return

    # Normalize row lines by taking the median offset modulo row height (53.45px).
    centers = [numpy.fmod(y, 53.45) for y in y_lines]
    centroid = round(numpy.median(centers))
    y_positions = numpy.arange(centroid, 480, 53.45).astype(int)

    for y in y_positions:
        if y < 40:
            continue  # Skip rows that are offscreen

        # Cut row slightly below and above the dashed line.
        row = frame[y - 40:y - 5, :]

        # Skip items that are not for sale (price region is lighter).
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
    return bool(diff.mean() < 4)


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
    """Detect when the user is not scrolling in a consistent direction (ignoring +-2px jitter)."""
    scroll_deltas = typing.cast(numpy.ndarray, numpy.diff(scroll_positions))
    downscroll_count = numpy.count_nonzero(scroll_deltas > 2)
    upscroll_count = numpy.count_nonzero(scroll_deltas < -2)
    return bool(downscroll_count > 10 and upscroll_count > 10)


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
    del lang  # LSTM engine ignores legacy Tesseract 3 CJK parameters.
    return '--psm 6 -c preserve_interword_spaces=1 -c tessedit_do_invert=0'


def _cleanup_name(item_name: str, lang: str) -> str:
    """Applies manual name cleanup and Unicode normalization to fix common OCR artifacts."""
    item_name = item_name.strip()
    item_name = item_name.replace('’', "'").replace('‘', "'")
    item_name = item_name.replace('Ao dai', 'Áo dài')
    item_name = _BAIL_RE.sub(r'\1all', item_name)

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
    """Fetches the item database set for a given locale, with caching."""
    with open(f'items/{locale}.json', encoding='utf-8') as fp:
        return set(json.load(fp))


@functools.lru_cache(maxsize=None)
def _get_item_list(locale: str) -> List[str]:
    """Fetches a sorted list of items (plus legacy aliases) for rapidfuzz matching."""
    items = set(_get_item_db(locale))
    items.update(LEGACY_ALIASES.get(locale, {}).keys())
    return sorted(items)


def _detect_locale(item_rows: List[numpy.ndarray], locale: str) -> str:
    """Deterministically detects the catalog locale by scoring candidate OCR outputs against item DBs."""
    if locale != 'auto':
        return locale

    # Deterministically pick up to 40 evenly spaced rows across the video.
    sample_size = min(40, len(item_rows))
    indices = numpy.linspace(0, len(item_rows) - 1, sample_size, dtype=int)
    sample_rows = [item_rows[int(i)] for i in indices]
    sample_image = Image.fromarray(cv2.vconcat(sample_rows))
    config = _get_tesseract_config('eng')

    def _score_candidate(lang: str) -> Tuple[Dict[str, int], Dict[str, Set[str]]]:
        try:
            parsed = pytesseract.image_to_string(sample_image, lang=lang, config=config)
        except pytesseract.TesseractError:
            if lang == 'script/Latin':
                parsed = pytesseract.image_to_string(sample_image, lang='eng', config=config)
            else:
                return {}, {}
        names = {_cleanup_name(line, lang) for line in parsed.split('\n')} - {''}
        exact_scores: Dict[str, int] = {}
        parsed_by_loc: Dict[str, Set[str]] = {}
        for loc in DETECTION_CANDIDATES[lang]:
            db = _get_item_db(loc)
            aliases = LEGACY_ALIASES.get(loc, {})
            exact_scores[loc] = sum(1 for n in names if aliases.get(n, n) in db)
            parsed_by_loc[loc] = names
        return exact_scores, parsed_by_loc

    exact_scores: Dict[str, int] = {}
    parsed_by_loc: Dict[str, Set[str]] = {}
    workers = min(len(DETECTION_CANDIDATES), max(2, MAX_OCR_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for scores, names_map in executor.map(_score_candidate, DETECTION_CANDIDATES):
            exact_scores.update(scores)
            parsed_by_loc.update(names_map)

    assert exact_scores, 'Failed to automatically detect language.'
    best_locale = max(
        exact_scores,
        key=lambda loc: (exact_scores[loc], -LOCALE_TIEBREAK_ORDER.index(loc)),
    )

    # Fallback to high-confidence fuzzy hits on tiny/noisy clips where exact hits are 0.
    if exact_scores[best_locale] == 0:
        fuzzy_scores: Dict[str, int] = {}
        for loc, names in parsed_by_loc.items():
            if not names:
                fuzzy_scores[loc] = 0
                continue
            item_list = _get_item_list(loc)
            fuzzy_scores[loc] = sum(
                1 for n in names
                if len(n) >= 3 and bool(process.extractOne(n, item_list, scorer=fuzz.ratio, score_cutoff=80.0))
            )
        best_locale = max(
            fuzzy_scores,
            key=lambda loc: (fuzzy_scores[loc], -LOCALE_TIEBREAK_ORDER.index(loc)),
        )
        assert fuzzy_scores[best_locale] > 0, 'Failed to automatically detect language.'

    logging.info('Detected locale: %s (exact=%d/%d)', best_locale, exact_scores[best_locale], sample_size)
    return best_locale


if __name__ == '__main__':
    results = scan('examples/catalog.mp4')
    print('\n'.join(results.items))
