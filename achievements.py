from common import ScanMode, ScanResult, color_match, slice_720p, to_720p

import functools
import json
import logging
import os
import re

import cv2
import numpy
from rapidfuzz import fuzz, process

import catalog

from typing import Any, Dict, Iterator, List, Optional, Tuple

# Periwinkle background and cream Nook Miles+ header (BGR).
BG_COLOR = (243, 207, 200)
TOP_COLOR = (220, 243, 251)

# Fixed 3-column horizontal bounds at 720p.
CARD_COLUMNS: Tuple[Tuple[int, int], ...] = (
    (54, 418),
    (445, 809),
    (836, 1200),
)

# Normalized (rx, ry) stamp centers inside a card for each max_tier (1..6).
STAMP_SLOTS: Dict[int, Tuple[Tuple[float, float], ...]] = {
    1: ((0.50, 0.66),),
    2: ((0.34, 0.66), (0.66, 0.66)),
    3: ((0.22, 0.68), (0.50, 0.68), (0.78, 0.68)),
    4: ((0.22, 0.78), (0.41, 0.54), (0.59, 0.78), (0.78, 0.54)),
    5: ((0.17, 0.76), (0.34, 0.53), (0.50, 0.76), (0.66, 0.53), (0.83, 0.76)),
    6: ((0.14, 0.77), (0.28, 0.54), (0.42, 0.77), (0.57, 0.54), (0.71, 0.77), (0.86, 0.54)),
}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ACHIEVEMENTS_PATH = os.path.join(_BASE_DIR, 'achievements', 'names.json')
_ISLAND_MILES_RE = re.compile(r'^.+\s+Miles!?$', re.IGNORECASE)
_ISLAND_ICONS_RE = re.compile(r'^.+\s+Icons$', re.IGNORECASE)


@functools.lru_cache(maxsize=1)
def _get_achievement_db() -> Tuple[List[str], Dict[str, int]]:
    with open(_ACHIEVEMENTS_PATH, encoding='utf-8') as fp:
        data = json.load(fp)
    names = [row['name'] for row in data]
    tiers = {row['name']: int(row['tiers']) for row in data}
    return names, tiers


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing the Nook Miles Achievements phone app."""
    side_color = slice_720p(frame, 150, 160, -20, None).mean(axis=(0, 1))
    if not color_match(side_color, BG_COLOR, 20):
        return False
    top_color = slice_720p(frame, None, 20, -20, None).mean(axis=(0, 1))
    return bool(color_match(top_color, TOP_COLOR, 22))


def scan(video_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans a video or screenshot of Nook Miles achievements and returns matched achievements + tier progress."""
    resolved_locale = 'en-us' if locale == 'auto' else locale
    cards = parse_video(video_file)
    details, unmatched = match_achievements(cards, locale=resolved_locale)
    items = [d['name'] for d in details]

    return ScanResult(
        mode=ScanMode.ACHIEVEMENTS,
        items=items,
        locale=resolved_locale,
        unmatched=unmatched,
        details=details,
    )


def parse_video(filename: str) -> List[numpy.ndarray]:
    """Parses a video/image and returns deduplicated full achievement card crops."""
    kept: List[Tuple[int, numpy.ndarray, numpy.ndarray, bool]] = []  # (col, card_bgr, thumb32x20, is_hovered)

    for frame in _read_frames(filename):
        for col_idx, card, is_hovered in _parse_frame(frame):
            thumb = cv2.resize(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY), (32, 20), interpolation=cv2.INTER_AREA)
            dup_idx: Optional[int] = None
            for i in range(len(kept) - 1, max(-1, len(kept) - 18), -1):
                k_col, _, k_thumb, _ = kept[i]
                if k_col != col_idx:
                    continue
                if float(cv2.absdiff(thumb, k_thumb).mean()) < 9.0:
                    dup_idx = i
                    break
            if dup_idx is not None:
                # Prefer unhovered card over hovered/tilted card.
                if kept[dup_idx][3] and not is_hovered:
                    kept[dup_idx] = (col_idx, card, thumb, is_hovered)
            else:
                kept.append((col_idx, card, thumb, is_hovered))

    cards = [c for _, c, _, _ in kept]
    assert cards, 'No Nook Miles achievements found, invalid video?'
    return cards


def _prep_title_row(card: numpy.ndarray) -> Optional[numpy.ndarray]:
    """Extracts and normalizes the achievement card title into a 35x415 grayscale strip for AcnhNanoOcrNet."""
    h, w = card.shape[:2]
    ty1, ty2 = int(h * 0.11), int(h * 0.29)
    tx1, tx2 = int(w * 0.05), int(w * 0.95)
    title_gray = cv2.cvtColor(card[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY)
    # Invert dark-background cards (e.g. Wispy Island Secrets night-sky card with light text).
    if float(numpy.median(title_gray)) < 140.0:
        title_gray = (255 - title_gray).astype(numpy.uint8)
    # Lift pastel card background cleanly to 245 while keeping dark brown title ink crisp.
    p85 = float(numpy.percentile(title_gray, 85))
    lo = min(145.0, p85 - 25.0)
    scale = 105.0 / max(15.0, p85 - lo)
    t_f = title_gray.astype(numpy.float32)
    norm = numpy.where(t_f <= lo, t_f, lo + (t_f - lo) * scale)
    norm = numpy.clip(norm, 0, 245).astype(numpy.uint8)

    cols_ink = numpy.nonzero((norm < 180).any(axis=0))[0]
    if len(cols_ink) < 4:
        return None
    left = max(0, int(cols_ink[0]) - 6)
    right = min(norm.shape[1], int(cols_ink[-1]) + 8)
    crop = norm[:, left:right]

    # Preserve aspect ratio when scaling to 35px height (matches AcnhNanoOcrNet training aspect ratio).
    th, tw = crop.shape[:2]
    target_h = 30
    scaled_w = min(410, max(16, round(tw * (target_h / max(1, th)))))
    resized = cv2.resize(crop, (scaled_w, target_h), interpolation=cv2.INTER_AREA)
    strip = numpy.full((35, 415), 245, dtype=numpy.uint8)
    strip[3:3 + target_h, 2:2 + scaled_w] = resized
    return strip


def count_filled_stamps(card: numpy.ndarray, max_tier: int) -> int:
    """Counts filled (colored/dated) stamp slots vs blank cream-white circles on an achievement card."""
    slots = STAMP_SLOTS.get(max_tier)
    if not slots:
        return 1
    h, w = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    half = max(10, int(min(w, h) * 0.065))

    filled = 0
    for rx, ry in slots:
        cx, cy = int(rx * w), int(ry * h)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        patch_g = gray[y1:y2, x1:x2]
        patch_s = hsv[y1:y2, x1:x2, 1]
        if patch_g.size == 0:
            continue
        # Blank circles are flat bright cream (p10 > 225, mean_sat < 22); filled stamps have dark ink or rich color.
        p10_gray = float(numpy.percentile(patch_g, 10))
        mean_sat = float(patch_s.mean())
        if p10_gray < 195.0 or mean_sat > 35.0:
            filled += 1
    return max(1, min(max_tier, filled))


def _match_title(raw_text: str, ach_names: List[str], ach_set: set) -> Optional[str]:
    clean = raw_text.strip()
    if not clean:
        return None
    if clean in ach_set:
        return clean
    if _ISLAND_MILES_RE.match(clean):
        return '(island name) Miles!'
    if _ISLAND_ICONS_RE.match(clean):
        return '(island name) Icons'

    # Fuzzy suffix check for "<Island Name> Miles!" and "<Island Name> Icons".
    parts = clean.split()
    if len(parts) >= 2:
        last = parts[-1].lower()
        if fuzz.ratio(last, 'miles!') >= 60.0 or fuzz.ratio(last, 'miles') >= 65.0:
            return '(island name) Miles!'
        if fuzz.ratio(last, 'icons') >= 60.0:
            return '(island name) Icons'

    # Strip trailing 1-2 char icon artifacts (e.g. Bug-Off! bug icon / Fishing Tourney! fish icon).
    trimmed = re.sub(r'([!?])\s+\S{1,3}$', r'\1', clean)
    if trimmed in ach_set:
        return trimmed

    cands = process.extract(trimmed, ach_names, scorer=fuzz.ratio, limit=2, score_cutoff=62.0)
    if not cands:
        return None
    best_name, best_score, _ = cands[0]
    second_score = cands[1][1] if len(cands) > 1 else 0.0
    if best_score >= 80.0 or (best_score - second_score) >= 8.0:
        return best_name
    return None


def match_achievements(
    cards: List[numpy.ndarray],
    locale: str = 'en-us',
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """OCRs achievement card titles and counts completed stamp tiers."""
    ach_names, ach_tiers = _get_achievement_db()
    ach_set = set(ach_names)

    valid_cards: List[numpy.ndarray] = []
    title_rows: List[numpy.ndarray] = []
    for card in cards:
        row = _prep_title_row(card)
        if row is not None:
            valid_cards.append(card)
            title_rows.append(row)

    if not title_rows:
        return [], []

    ocr = catalog._get_nano_ocr()
    allowed = ocr.allowed_by_locale.get(locale, ocr.allowed_by_locale['en-us'])
    prepped, ink_ws = ocr.prep_rows(title_rows)

    raw_texts: List[str] = []
    batch_size = 64
    for start in range(0, len(title_rows), batch_size):
        end = min(len(title_rows), start + batch_size)
        max_w = int(ink_ws[start:end].max())
        w_b = min(384, max(128, ((max_w + 15) // 16) * 16))
        x_b = (prepped[start:end, None, :, :w_b].astype(numpy.float32) - 127.5) * (1.0 / 127.5)
        logits = ocr.sess.run(None, {'images': x_b})[0]
        decoded = ocr.decode_masked(logits, allowed, lang='eng')
        raw_texts.extend(decoded)

    best_tiers: Dict[str, int] = {}
    unmatched: List[str] = []
    for card, raw in zip(valid_cards, raw_texts):
        matched_name = _match_title(raw, ach_names, ach_set)
        if not matched_name:
            # Skip locked/unachieved cards ("???" / blank) without counting as errors.
            if set(raw.strip()) <= {'?', '.', ' ', '-'}:
                continue
            logging.warning('Unmatched achievement title: %r', raw)
            unmatched.append(raw)
            continue
        max_tier = ach_tiers[matched_name]
        tier = count_filled_stamps(card, max_tier)
        if matched_name not in best_tiers or tier > best_tiers[matched_name]:
            best_tiers[matched_name] = tier

    details = [
        {'name': name, 'tier': tier, 'max_tier': ach_tiers[name]}
        for name, tier in sorted(best_tiers.items())
    ]
    total_seen = len(details) + len(unmatched)
    if not details or (total_seen >= 6 and len(unmatched) > max(5, int(0.45 * total_seen))):
        raise AssertionError('Failed to match achievements — currently only English Nook Miles achievements are supported.')
    return details, unmatched


def _read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Yields 720p-normalized frames showing Nook Miles Achievements."""
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(4, round(fps / 7.5)) if fps >= 45.0 else 4
    seen_valid = False
    try:
        while True:
            if seen_valid and frame_step > 1:
                eof = False
                for _ in range(frame_step - 1):
                    if not cap.grab():
                        eof = True
                        break
                if eof:
                    break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if not detect(frame):
                continue
            seen_valid = True
            yield to_720p(frame)
    finally:
        cap.release()


def _parse_frame(frame: numpy.ndarray) -> Iterator[Tuple[int, numpy.ndarray, bool]]:
    """Extracts full achievement cards independently per column to handle hovered card overhang."""
    bg_arr = numpy.array(BG_COLOR, dtype=numpy.float32)
    for col_idx, (x1, x2) in enumerate(CARD_COLUMNS):
        col = frame[115:700, x1:x2]
        dist = numpy.linalg.norm(col.astype(numpy.float32) - bg_arr, axis=2)
        bg_row = (dist < 28.0).mean(axis=1) > 0.70
        non_bg = numpy.nonzero(~bg_row)[0] + 115
        if len(non_bg) < 50:
            continue
        splits = numpy.nonzero(numpy.diff(non_bg) > 5)[0]
        prev = 0
        for s in list(splits) + [len(non_bg) - 1]:
            y1, y2 = int(non_bg[prev]), int(non_bg[s])
            prev = s + 1
            h = y2 - y1
            if 198 <= h <= 242:
                is_hovered = h > 224
                yield col_idx, frame[y1:y2, x1:x2], is_hovered
