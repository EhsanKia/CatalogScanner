from common import ScanMode, ScanResult, crop_720p_roi, slice_720p

import concurrent.futures
import contextlib
import cv2
import dataclasses
import functools
import json
import logging
import math
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


def _available_cpus() -> int:
    """Returns usable CPU count, respecting Linux cgroup/affinity limits when available."""
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    return os.cpu_count() or 2


# Chunk size bounds per Tesseract call (900 rows * 35px = 31.5k, safely below Tesseract's 32k height limit).
MIN_OCR_CHUNK_ROWS = 250
MAX_OCR_CHUNK_ROWS = 900
MAX_OCR_WORKERS = min(4, _available_cpus())

ROW_TRACK_WINDOW = 24
ROW_DEDUPE_MEAN_THRESH = 7.5
ROW_DEDUPE_BLOCK_THRESH = 16.0

TESSERACT_CONFIG = '--psm 6 -c preserve_interword_spaces=1 -c tessedit_do_invert=0'

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
_EN_LEGACY_ALIASES: Dict[str, str] = {
    'Writing desk': 'Study desk',
    'Writing chair': 'Study chair',
    'Writing poster': 'Study poster',
    'Teacup ride': 'Plaza teacup ride',
    'Fish and chips': 'Fish-and-chips',
    'Tee with silicon bib': 'Tee with silicone bib',
    'Beige desert-tile flooring': 'Beige Moroccan flooring',
    'Beige desert-tile wall': 'Beige Moroccan wall',
    'Blue desert-tile flooring': 'Blue Moroccan flooring',
    'Blue desert-tile wall': 'Blue Moroccan wall',
    'Olive desert-tile flooring': 'Olive Moroccan flooring',
    'Olive desert-tile wall': 'Olive Moroccan wall',
    'Purple desert-tile flooring': 'Purple Moroccan flooring',
    'Purple desert-tile wall': 'Purple Moroccan wall',
    'Paper bag': 'Paper-bag hood',
}
LEGACY_ALIASES: Dict[str, Dict[str, str]] = {
    'en-us': _EN_LEGACY_ALIASES,
    'en-eu': _EN_LEGACY_ALIASES,
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
    side_color = slice_720p(frame, 150, 160, -20, None).mean(axis=(0, 1))
    if numpy.linalg.norm(side_color - WARDELL_COLOR) < 20:
        raise AssertionError('Wardell catalog is not supported.')
    if numpy.linalg.norm(side_color - NOOK_MILES_COLOR) < 20:
        raise AssertionError('Nook Miles catalog is not supported.')
    if numpy.linalg.norm(side_color - SIDE_COLOR) >= 22:
        return False
    top_color = slice_720p(frame, None, 20, -20, None).mean(axis=(0, 1))
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
    # Native 30fps Switch captures stay on each page for ~3 frames; scale for 60fps capture cards.
    if fps <= 0:
        frame_step = 3
    elif fps >= 29.0:
        frame_step = max(3, round(fps / 10.0))
    elif fps >= 20.0:
        frame_step = 2
    else:
        frame_step = 1

    unfinished_page = False
    prev_raw_rows: List[numpy.ndarray] = []
    tracks: List[_RowTrack] = []
    with contextlib.closing(_read_frames(cap, log_label=filename)) as frames:
        for i, frame in enumerate(frames):
            if not unfinished_page and i % frame_step != 0:
                continue  # Skip intermediate frames on the same page
            new_rows = list(_parse_frame(frame, for_sale))
            if not new_rows:
                continue
            if (
                not unfinished_page
                and len(prev_raw_rows) == len(new_rows) > 4
                and cv2.absdiff(cv2.vconcat(prev_raw_rows), cv2.vconcat(new_rows)).mean() < 2.5
            ):
                continue  # Skip identical page frames

            # Switch's font rendering sometimes lags fast page scrolling, leaving bottom rows blank.
            unfinished_page = any(r.min() > 150 for r in new_rows)
            prev_raw_rows = new_rows
            _track_frame_rows(tracks, new_rows)

    assert tracks, 'No items found, invalid video?'
    return [t.ref for t in tracks]


def _ocr_chunk(chunk: List[numpy.ndarray], lang: str) -> Set[str]:
    """Runs Tesseract OCR on a single chunk of vertically concatenated row strips."""
    parsed_text = pytesseract.image_to_string(
        Image.fromarray(cv2.vconcat(chunk)),
        lang=lang,
        config=TESSERACT_CONFIG,
    )
    assert isinstance(parsed_text, str), 'Tesseract returned bytes'
    return {_cleanup_name(item, lang) for item in parsed_text.split('\n')} - {''}


def run_ocr(item_rows: List[numpy.ndarray], lang: str = 'eng') -> Set[str]:
    """Runs Tesseract OCR across item rows in parallel chunks and returns all parsed lines."""
    if not item_rows:
        return set()

    n = len(item_rows)
    workers = min(MAX_OCR_WORKERS, math.ceil(n / MIN_OCR_CHUNK_ROWS))
    n_chunks = max(workers, math.ceil(n / MAX_OCR_CHUNK_ROWS))
    chunk_size = math.ceil(n / n_chunks)
    chunks = [item_rows[i:i + chunk_size] for i in range(0, n, chunk_size)]
    logging.debug('Running Tesseract (%s) on %d rows across %d chunk(s) (%d worker(s))', lang, n, len(chunks), workers)

    if workers <= 1:
        return set().union(*(_ocr_chunk(chunk, lang) for chunk in chunks))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return set().union(*executor.map(lambda chunk: _ocr_chunk(chunk, lang), chunks))


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

    if no_match_items:
        logging.warning('Failed to match %d items: %s', len(no_match_items), no_match_items)
        # Allow a floor of 3 unmatched items on tiny catalogs so 1-3 OCR glitches don't abort them.
        assert len(no_match_items) <= max(3, int(0.3 * len(item_names))), \
            'Failed to match multiple items, wrong language?'

    return sorted(matched_items), no_match_items


def _read_frames(
    source: Union[str, cv2.VideoCapture],
    log_label: Optional[str] = None,
) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the catalog list region in grayscale."""
    cap = cv2.VideoCapture(source) if isinstance(source, str) else source
    label = log_label or (source if isinstance(source, str) else '<video>')
    seen_valid = False
    pending_gap = 0
    total_gap_frames = 0
    scroll_positions: List[int] = []
    try:
        while True:
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
                if seen_valid:
                    pending_gap += 1
                continue

            gap_before = pending_gap
            if pending_gap:
                total_gap_frames += pending_gap
                pending_gap = 0
            seen_valid = True

            if gap_before >= 5:
                scroll_positions = []  # Reset scroll positions on catalog section change.

            # Crop scrollbar region and get scroll position, then warn about bad scrolling.
            scrollbar = cv2.cvtColor(
                crop_720p_roi(frame, 160, 570, 1235, 1245),
                cv2.COLOR_BGR2GRAY,
            ).mean(axis=1)
            dark = scrollbar < 150
            if dark.any():
                scroll_positions.append(int(numpy.argmax(dark)))
                if _is_inconsistent_scroll(scroll_positions):
                    raise AssertionError('Video is scrolling inconsistently.')

            # Crop only the item name and price ROI before resizing/grayscaling.
            yield cv2.cvtColor(
                crop_720p_roi(frame, 150, 630, 635, 1220),
                cv2.COLOR_BGR2GRAY,
            )
    finally:
        cap.release()
        if total_gap_frames:
            logging.warning('Skipped %d non-matching mid-video frames in %s', total_gap_frames, label)


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


@dataclasses.dataclass
class _RowTrack:
    ref: numpy.ndarray
    ink_width: int
    is_hovered: bool
    sharpness: float


def _normalize_row(row: numpy.ndarray) -> Tuple[numpy.ndarray, bool]:
    """Lifts the dark diagonal hover highlight background to white while preserving dark glyph ink."""
    if float(numpy.percentile(row, 90)) >= 235.0:
        return row, False
    right = row[:, 350:415]
    stripe_lo = float(numpy.percentile(right, 5)) if right.min() > 170 else 200.0
    stripe_lo = max(190.0, min(215.0, stripe_lo))
    scale = 105.0 / max(10.0, stripe_lo - 140.0)
    row_f = row.astype(numpy.float32)
    stretched = numpy.where(row_f <= 140.0, row_f, 140.0 + (row_f - 140.0) * scale)
    return numpy.clip(stretched, 0, 245).astype(numpy.uint8), True


def _ink_width(row: numpy.ndarray) -> int:
    """Returns the padded horizontal extent of dark glyph ink inside a normalized row."""
    cols = numpy.nonzero((row < 180).any(axis=0))[0]
    return min(row.shape[1], int(cols[-1]) + 4) if len(cols) else 0


def _row_sharpness(row: numpy.ndarray, width: int) -> float:
    """Estimates glyph edge sharpness via mean L1 gradient within the inked region."""
    if width < 2 or row.shape[0] <= 6:
        return 0.0
    sub = row[3:-3, :width].astype(numpy.int16)
    return float(numpy.abs(numpy.diff(sub, axis=1)).mean() + numpy.abs(numpy.diff(sub, axis=0)).mean())


def _align_row_diff(a: numpy.ndarray, b: numpy.ndarray, width: int) -> Tuple[float, float, int]:
    """Computes minimum (mean_diff, max_16px_block_diff, dy) across small vertical alignments."""
    best_mean, best_blk, best_dy = 999.0, 999.0, 0
    h = a.shape[0]
    a_mid = a[2:-2, :width]
    n_blk = width // 16
    w_trim = n_blk * 16
    for dy in (0, -1, 1, -2, 2):
        diff = cv2.absdiff(a_mid, b[2 + dy:h - 2 + dy, :width])
        mean_diff = float(diff.mean())
        if mean_diff < best_mean:
            if w_trim > 0:
                blk_diff = float(diff[:, :w_trim].reshape(diff.shape[0], n_blk, 16).mean(axis=(0, 2)).max())
                if width > w_trim:
                    blk_diff = max(blk_diff, float(diff[:, w_trim:].mean()))
            else:
                blk_diff = mean_diff
            best_mean, best_blk, best_dy = mean_diff, blk_diff, dy
            if mean_diff < 3.0 and blk_diff < 10.0:
                break
    return best_mean, best_blk, best_dy


def _track_frame_rows(tracks: List[_RowTrack], new_rows: List[numpy.ndarray]) -> None:
    """Merges normalized non-blank rows into recent tracks, keeping the sharpest unhovered crop."""
    start = max(0, len(tracks) - ROW_TRACK_WINDOW)
    for raw in new_rows:
        if raw.min() > 150:
            continue
        norm, hov = _normalize_row(raw)
        w = _ink_width(norm)
        if w < 8:
            continue
        sharp = _row_sharpness(norm, w)

        matched_track: Optional[_RowTrack] = None
        best_mean = ROW_DEDUPE_MEAN_THRESH
        for ti in range(len(tracks) - 1, start - 1, -1):
            t = tracks[ti]
            if abs(t.ink_width - w) > 8:
                continue
            mean_diff, blk_diff, _ = _align_row_diff(t.ref, norm, max(t.ink_width, w))
            if mean_diff < best_mean and blk_diff < ROW_DEDUPE_BLOCK_THRESH:
                best_mean = mean_diff
                matched_track = t
                if mean_diff < 3.0:
                    break

        if matched_track is not None:
            if (matched_track.is_hovered and not hov) or (
                (not hov or matched_track.is_hovered) and sharp > matched_track.sharpness * 1.12
            ):
                matched_track.ref = norm
                matched_track.ink_width = w
                matched_track.is_hovered = hov
                matched_track.sharpness = sharp
        else:
            tracks.append(_RowTrack(
                ref=norm,
                ink_width=w,
                is_hovered=hov,
                sharpness=sharp,
            ))


def _is_inconsistent_scroll(scroll_positions: List[int]) -> bool:
    """Detect when the user is not scrolling in a consistent direction (ignoring +-2px jitter)."""
    scroll_deltas = typing.cast(numpy.ndarray, numpy.diff(scroll_positions))
    downscroll_count = numpy.count_nonzero(scroll_deltas > 2)
    upscroll_count = numpy.count_nonzero(scroll_deltas < -2)
    return bool(downscroll_count > 10 and upscroll_count > 10)


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
    sample_array = cv2.vconcat([item_rows[int(i)] for i in indices])

    def _score_candidate(lang: str) -> Tuple[Dict[str, int], Dict[str, Set[str]]]:
        image = Image.fromarray(sample_array)
        try:
            parsed = pytesseract.image_to_string(image, lang=lang, config=TESSERACT_CONFIG)
        except pytesseract.TesseractError as err:
            if lang == 'script/Latin':
                logging.warning('Tesseract language %r unavailable (%s); falling back to eng', lang, err)
                try:
                    parsed = pytesseract.image_to_string(image, lang='eng', config=TESSERACT_CONFIG)
                except pytesseract.TesseractError as fallback_err:
                    logging.warning('Tesseract fallback lang eng unavailable: %s', fallback_err)
                    return {}, {}
            else:
                logging.warning('Tesseract language %r unavailable: %s', lang, err)
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

    # Score Latin first: most scans are Latin, avoiding 5 extra CJK/Cyrillic OCR passes.
    exact_scores, parsed_by_loc = _score_candidate('script/Latin')
    fast_threshold = max(min(5, sample_size), int(0.4 * sample_size))
    if exact_scores:
        best_latin = max(
            exact_scores,
            key=lambda loc: (exact_scores[loc], -LOCALE_TIEBREAK_ORDER.index(loc)),
        )
        if exact_scores[best_latin] >= fast_threshold:
            logging.info(
                'Detected locale: %s (exact=%d/%d, latin fast-path)',
                best_latin, exact_scores[best_latin], sample_size,
            )
            return best_latin

    other_langs = [lang for lang in DETECTION_CANDIDATES if lang != 'script/Latin']
    workers = min(len(other_langs), max(2, MAX_OCR_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for scores, names_map in executor.map(_score_candidate, other_langs):
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
