from common import ScanMode, ScanResult, circular_phase, color_match, crop_720p_roi, slice_720p

import collections
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
import threading
import typing
import unicodedata

from PIL import Image
from rapidfuzz import fuzz, process
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple, Union

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

CJK_LOCALES: FrozenSet[str] = frozenset({'ja-jp', 'zh-cn', 'zh-tw', 'ko-kr'})


def _build_confusion_map(groups: Iterable[Iterable[str]]) -> Dict[str, FrozenSet[str]]:
    """Builds a symmetric char -> confusable-peers lookup from character pairs or groups."""
    mapping: Dict[str, Set[str]] = collections.defaultdict(set)
    for group in groups:
        chars = tuple(group)
        for ch in chars:
            mapping[ch].update(other for other in chars if other != ch)
    return {ch: frozenset(peers) for ch, peers in mapping.items()}


_JA_SMALL_TO_LARGE = str.maketrans(
    'ぁぃぅぇぉっゃゅょゎァィゥェォヵヶッャュョヮ',
    'あいうえおつやゆよわアイウエオカケツヤユヨワ',
)
_JA_SMALL_CHARS: FrozenSet[str] = frozenset('ぁぃぅぇぉっゃゅょゎァィゥェォヵヶッャュョヮ')
_JA_VISUAL_CONFUSIONS = _build_confusion_map([
    ('シ', 'ツ', 'ン'), ('ソ', 'ン'), ('ロ', '口'), ('カ', '力'),
    ('ニ', '二', '三'), ('エ', '工'), ('ハ', '八'), ('オ', '才'),
    ('ー', '一'), ('ー', '二'), ('ー', 'ニ'), ('ー', '三'),
])
_ZH_VISUAL_CONFUSIONS = _build_confusion_map([
    ('五', '石'), ('蓝', '篮'), ('圖', '圓'), ('图', '圆'), ('晴', '蜻'),
    ('春', '素'), ('決', '灑'), ('沁', '灑'), ('綢', '繩'), ('繹', '繩'), ('绸', '绳'),
    ('穴', '窯'), ('窗', '窑'), ('窒', '窑'), ('摇', '搖'), ('乌', '鸟'),
])
_KO_JAMO_CONFUSIONS = _build_confusion_map([
    'ㄱㄲㅋ', 'ㄷㄸㅌ', 'ㅂㅃㅍ', 'ㅅㅆㅊ', 'ㅈㅉㅊ',
    'ㅏㅑ', 'ㅓㅕ', 'ㅓㅔ', 'ㅗㅛ', 'ㅜㅠ', 'ㅐㅔ',
])

_CHOSEONG = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
_JUNGSEONG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
_JONGSEONG = (
    '', 'ㄱ', 'ㄲ', 'ㄱㅅ', 'ㄴ', 'ㄴㅈ', 'ㄴㅎ', 'ㄷ', 'ㄹ', 'ㄹㄱ', 'ㄹㅁ', 'ㄹㅂ',
    'ㄹㅅ', 'ㄹㅌ', 'ㄹㅍ', 'ㄹㅎ', 'ㅁ', 'ㅂ', 'ㅂㅅ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
)
_HANGUL_JAMO_TO_COMPAT: Dict[str, str] = {
    **{chr(0x1100 + i): c for i, c in enumerate(_CHOSEONG)},
    **{chr(0x1161 + i): c for i, c in enumerate(_JUNGSEONG)},
    **{chr(0x11A7 + i): c for i, c in enumerate(_JONGSEONG) if c},
}
_PROLONGED_STUTTER_RE = re.compile(r'ー{2,}')

def _load_legacy_aliases() -> Dict[str, Dict[str, str]]:
    """Loads per-locale legacy item rename mappings from items/legacy_aliases.json."""
    alias_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'items', 'legacy_aliases.json')
    with open(alias_path, encoding='utf-8') as fp:
        return json.load(fp)


# Items renamed across AC:NH patches (1.3 -> 3.0) mapped to current canonical names.
LEGACY_ALIASES: Dict[str, Dict[str, str]] = _load_legacy_aliases()

# Cyrillic <-> Latin homoglyphs (plus visual near-twin 'п' -> 'n') for mixed-script Russian items.
_RU_HOMOGLYPH_TRANS = str.maketrans('АВЕКМНОРСТУХаеорсухп', 'ABEKMHOPCTYXaeopcyxn')

# Unified 14-locale Nano CTC OCR model parameters.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_NANO_ONNX_PATH = os.path.join(_BASE_DIR, 'models', 'acnh_nano_fp16.onnx')
_NANO_VOCAB_PATH = os.path.join(_BASE_DIR, 'models', 'acnh_nano_vocab.json')
_NANO_TARGET_H = 32
_NANO_TARGET_W = 384
_NANO_BATCH_SIZE = 64
_OCR_TLS = threading.local()

_LANG_TO_LOCALES: Dict[str, Tuple[str, ...]] = {
    'deu': ('de-eu',),
    'eng': ('en-us', 'en-eu'),
    'spa': ('es-eu', 'es-us'),
    'fra': ('fr-eu', 'fr-us'),
    'ita': ('it-eu',),
    'jpn': ('ja-jp',),
    'kor': ('ko-kr',),
    'nld': ('nl-eu',),
    'rus': ('ru-eu',),
    'chi_sim': ('zh-cn',),
    'chi_tra': ('zh-tw',),
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
    if color_match(side_color, WARDELL_COLOR, 20):
        raise AssertionError('Wardell catalog is not supported.')
    if color_match(side_color, NOOK_MILES_COLOR, 20):
        raise AssertionError('Nook Miles catalog is not supported.')
    if not color_match(side_color, SIDE_COLOR, 22):
        return False
    top_color = slice_720p(frame, None, 20, -20, None).mean(axis=(0, 1))
    return color_match(top_color, TOP_COLOR, 45)


def scan(video_file: str, locale: str = 'en-us', for_sale: bool = False) -> ScanResult:
    """Scans a video of scrolling through a catalog and returns all items found."""
    item_rows = parse_video(video_file, for_sale)
    locale = _detect_locale(item_rows, locale)
    item_names = run_ocr(item_rows, lang=LOCALE_MAP[locale], locale=locale)
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
    # step_state = [frame_step, force_every_frame (0 or 1)]
    step_state = [frame_step, 0]
    with contextlib.closing(_read_frames(cap, log_label=filename, step_state=step_state)) as frames:
        for frame in frames:
            new_rows = list(_parse_frame(frame, for_sale))
            if not new_rows:
                step_state[1] = 0
                continue
            if (
                not unfinished_page
                and len(prev_raw_rows) == len(new_rows) > 4
                and cv2.absdiff(cv2.vconcat(prev_raw_rows), cv2.vconcat(new_rows)).mean() < 2.5
            ):
                step_state[1] = 0
                continue  # Skip identical page frames

            # Switch's font rendering sometimes lags fast page scrolling, leaving bottom rows blank.
            unfinished_page = any(r.min() > 150 for r in new_rows)
            step_state[1] = 1 if unfinished_page else 0
            prev_raw_rows = new_rows
            _track_frame_rows(tracks, new_rows)

    assert tracks, 'No items found, invalid video?'
    return [t.ref for t in tracks]


class _NanoOcrEngine:
    """Unified 14-locale CTC recognizer and single-pass locale detector backed by ONNX Runtime."""

    def __init__(self, onnx_path: str, vocab_path: str) -> None:
        import onnxruntime as ort

        with open(vocab_path, encoding='utf-8') as fp:
            vdata = json.load(fp)
        self.vocab: List[str] = vdata['vocab']
        self.locales: List[str] = vdata['locales']
        char_to_id = {ch: idx for idx, ch in enumerate(self.vocab) if idx > 0}

        self.allowed_by_locale: Dict[str, numpy.ndarray] = {}
        for loc in self.locales:
            db = _get_item_db(loc)
            aliases = LEGACY_ALIASES.get(loc, {})
            chars = set(''.join(db)) | set(''.join(aliases.keys()))
            self.allowed_by_locale[loc] = numpy.array(
                [0] + sorted(char_to_id[ch] for ch in chars if ch in char_to_id),
                dtype=numpy.int32,
            )

        self.allowed_by_lang: Dict[str, numpy.ndarray] = {}
        for lang_code, locs in _LANG_TO_LOCALES.items():
            merged = sorted({int(tok) for loc in locs for tok in self.allowed_by_locale[loc]})
            self.allowed_by_lang[lang_code] = numpy.array(merged, dtype=numpy.int32)

        cpus = _available_cpus()
        self.workers = min(4, cpus // 2) if cpus >= 4 else 1
        intra_threads = max(1, min(2, cpus // self.workers)) if self.workers > 1 else max(1, cpus)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_threads
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        self.sess.run(None, {'images': numpy.zeros((2, 1, _NANO_TARGET_H, 192), dtype=numpy.float32)})

    @staticmethod
    def prep_rows(rows: List[numpy.ndarray]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Crops each row to its ink width and resizes proportionally into 32x384 strips."""
        n = len(rows)
        prepped = numpy.empty((n, _NANO_TARGET_H, _NANO_TARGET_W), dtype=numpy.uint8)
        ink_ws = numpy.empty(n, dtype=numpy.int32)
        for i, row in enumerate(rows):
            cols = numpy.nonzero((row < 185).any(axis=0))[0]
            raw_w = min(row.shape[1], int(cols[-1]) + 13) if len(cols) else 35
            scaled_w = min(_NANO_TARGET_W, max(32, round(raw_w * _NANO_TARGET_W / row.shape[1])))
            prepped[i, :, :scaled_w] = cv2.resize(
                row[:, :raw_w], (scaled_w, _NANO_TARGET_H), interpolation=cv2.INTER_AREA,
            )
            if scaled_w < _NANO_TARGET_W:
                prepped[i, :, scaled_w:] = 245
            ink_ws[i] = scaled_w
        return prepped, ink_ws

    @staticmethod
    def _is_exact_db_hit(name: str, recover_locs: Tuple[str, ...]) -> bool:
        clean_key = name.replace('\u2060', '')
        for loc in recover_locs:
            db = _get_item_db(loc)
            aliases = LEGACY_ALIASES.get(loc, {})
            if aliases.get(name, name) in db or aliases.get(name, name).replace('\u2060', '') in _get_clean_db_map(loc):
                return True
            if clean_key in _get_clean_db_map(loc):
                return True
        return False

    def decode_masked(
        self,
        logits: numpy.ndarray,
        allowed: numpy.ndarray,
        lang: str = 'eng',
        recover_locs: Optional[Tuple[str, ...]] = None,
    ) -> List[str]:
        """Greedy CTC decode restricted to allowed tokens, with top-k logit recovery for 2-char DB misses."""
        sub_logits = logits[:, :, allowed]
        top1_pos = sub_logits.argmax(axis=-1)
        pred_ids = allowed[top1_pos]
        out: List[str] = []
        vocab = self.vocab
        for row_i, seq in enumerate(pred_ids):
            chars: List[str] = []
            char_steps: List[Tuple[int, int]] = []
            prev = 0
            for t_idx, tok in enumerate(seq):
                if tok != 0 and tok != prev:
                    chars.append(vocab[tok])
                    char_steps.append((t_idx, int(top1_pos[row_i, t_idx])))
                prev = tok
            if not chars:
                continue
            txt = ''.join(chars)
            if recover_locs and len(chars) == 2 and not self._is_exact_db_hit(_cleanup_name(txt, lang), recover_locs):
                row_l = sub_logits[row_i]
                row_exp = numpy.exp(row_l - row_l.max(axis=-1, keepdims=True))
                probs = row_exp / row_exp.sum(axis=-1, keepdims=True)
                best_prob = 0.0
                best_txt = txt
                for char_pos, (t_idx, pos1) in enumerate(char_steps):
                    top_k = numpy.argsort(-probs[t_idx])[:4]
                    for pos2 in top_k:
                        pos2_i = int(pos2)
                        tok2 = int(allowed[pos2_i])
                        if tok2 == 0 or pos2_i == pos1:
                            continue
                        p2 = float(probs[t_idx, pos2_i])
                        if p2 < 0.01:
                            break
                        alt_chars = list(chars)
                        alt_chars[char_pos] = vocab[tok2]
                        alt_txt = ''.join(alt_chars)
                        if p2 > best_prob and self._is_exact_db_hit(_cleanup_name(alt_txt, lang), recover_locs):
                            best_prob = p2
                            best_txt = alt_txt
                txt = best_txt
            out.append(txt)
        return out

    def detect_locale(self, item_rows: List[numpy.ndarray]) -> str:
        """Scores up to 40 evenly spaced rows across all 14 locales in a single ONNX pass."""
        assert item_rows, 'Failed to automatically detect language.'
        prepped, ink_ws = self.prep_rows(item_rows)
        sample_size = min(40, len(item_rows))
        s_idx = numpy.linspace(0, len(item_rows) - 1, sample_size, dtype=int)
        w_s = min(_NANO_TARGET_W, max(128, ((int(ink_ws[s_idx].max()) + 15) // 16) * 16))
        x_s = (prepped[s_idx, None, :, :w_s].astype(numpy.float32) - 127.5) * (1.0 / 127.5)
        s_logits = self.sess.run(None, {'images': x_s})[0]

        exact_scores: Dict[str, int] = {}
        decoded_by_loc: Dict[str, Set[str]] = {}
        for loc in LOCALE_TIEBREAK_ORDER:
            lang = LOCALE_MAP[loc]
            allowed = self.allowed_by_locale[loc]
            names = {_cleanup_name(txt, lang) for txt in self.decode_masked(s_logits, allowed) if txt} - {''}
            decoded_by_loc[loc] = names
            db = _get_item_db(loc)
            clean_db = _get_clean_db_map(loc)
            aliases = LEGACY_ALIASES.get(loc, {})
            exact_scores[loc] = sum(
                1 for n in names
                if aliases.get(n, n) in db or aliases.get(n, n).replace('\u2060', '') in clean_db
            )

        best_locale = max(
            exact_scores,
            key=lambda loc: (exact_scores[loc], -LOCALE_TIEBREAK_ORDER.index(loc)),
        )
        if exact_scores[best_locale] == 0:
            fuzzy_scores: Dict[str, int] = {}
            for loc, names in decoded_by_loc.items():
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

        _OCR_TLS.sample_cache = (item_rows[0], item_rows[-1], len(item_rows), prepped, ink_ws, s_idx, s_logits)
        logging.info('Detected locale: %s (exact=%d/%d, nano-onnx)', best_locale, exact_scores[best_locale], sample_size)
        return best_locale

    def recognize_rows(
        self,
        item_rows: List[numpy.ndarray],
        lang: str,
        locale: Optional[str] = None,
    ) -> Set[str]:
        """Recognizes all item rows using dynamic-width batching and reuses cached sample logits if available."""
        allowed = (
            self.allowed_by_locale[locale]
            if locale and locale in self.allowed_by_locale
            else self.allowed_by_lang[lang]
        )
        recover_locs = (locale,) if locale and locale in LOCALE_MAP else _LANG_TO_LOCALES[lang]
        cached = getattr(_OCR_TLS, 'sample_cache', None)
        _OCR_TLS.sample_cache = None

        names: Set[str] = set()
        if (
            cached is not None
            and len(item_rows) == cached[2]
            and item_rows[0] is cached[0]
            and item_rows[-1] is cached[1]
        ):
            _, _, _, prepped, ink_ws, s_idx, s_logits = cached
            for txt in self.decode_masked(s_logits, allowed, lang=lang, recover_locs=recover_locs):
                cleaned = _cleanup_name(txt, lang)
                if cleaned:
                    names.add(cleaned)
            rem_mask = numpy.ones(len(item_rows), dtype=bool)
            rem_mask[s_idx] = False
            rem_idx = numpy.nonzero(rem_mask)[0]
        else:
            prepped, ink_ws = self.prep_rows(item_rows)
            rem_idx = numpy.arange(len(item_rows))

        if len(rem_idx):
            order = rem_idx[numpy.argsort(ink_ws[rem_idx])]
            batches: List[numpy.ndarray] = []
            for start in range(0, len(order), _NANO_BATCH_SIZE):
                b_idx = order[start:start + _NANO_BATCH_SIZE]
                w_b = min(_NANO_TARGET_W, max(128, ((int(ink_ws[b_idx].max()) + 15) // 16) * 16))
                x_b = (prepped[b_idx, None, :, :w_b].astype(numpy.float32) - 127.5) * (1.0 / 127.5)
                batches.append(x_b)

            def _run_batch(xb: numpy.ndarray) -> List[str]:
                logits = self.sess.run(None, {'images': xb})[0]
                return self.decode_masked(logits, allowed, lang=lang, recover_locs=recover_locs)

            if self.pool is not None and len(batches) > 1:
                batch_outs = list(self.pool.map(_run_batch, batches))
            else:
                batch_outs = [_run_batch(xb) for xb in batches]

            for lines in batch_outs:
                for txt in lines:
                    cleaned = _cleanup_name(txt, lang)
                    if cleaned:
                        names.add(cleaned)

        return names


@functools.lru_cache(maxsize=1)
def _get_nano_ocr() -> Optional[_NanoOcrEngine]:
    """Loads and caches the unified 14-locale Nano FP16 ONNX OCR engine."""
    try:
        return _NanoOcrEngine(_NANO_ONNX_PATH, _NANO_VOCAB_PATH)
    except Exception:
        logging.warning('Nano ONNX OCR unavailable; falling back to Tesseract.', exc_info=True)
        return None


def _run_nano_ocr(
    item_rows: List[numpy.ndarray],
    lang: str,
    locale: Optional[str] = None,
) -> Optional[Set[str]]:
    """Runs unified Nano ONNX OCR across row crops; returns None on unavailability or error."""
    if lang not in _LANG_TO_LOCALES and (not locale or locale not in LOCALE_MAP):
        return None
    engine = _get_nano_ocr()
    if engine is None:
        return None
    try:
        logging.debug('Running Nano ONNX OCR (lang=%s, locale=%s) on %d rows', lang, locale, len(item_rows))
        return engine.recognize_rows(item_rows, lang=lang, locale=locale)
    except Exception:
        logging.warning('Nano ONNX OCR failed for %s; falling back to Tesseract.', lang, exc_info=True)
        return None


def _ocr_chunk(chunk: List[numpy.ndarray], lang: str) -> Set[str]:
    """Runs Tesseract OCR on a single chunk of vertically concatenated row strips."""
    parsed_text = pytesseract.image_to_string(
        Image.fromarray(cv2.vconcat(chunk)),
        lang=lang,
        config=TESSERACT_CONFIG,
    )
    assert isinstance(parsed_text, str), 'Tesseract returned bytes'
    return {_cleanup_name(item, lang) for item in parsed_text.split('\n')} - {''}


def run_ocr(
    item_rows: List[numpy.ndarray],
    lang: str = 'eng',
    locale: Optional[str] = None,
) -> Set[str]:
    """Runs unified Nano ONNX OCR across item rows with parallel Tesseract fallback."""
    if not item_rows:
        return set()

    nano_names = _run_nano_ocr(item_rows, lang=lang, locale=locale)
    if nano_names is not None:
        return nano_names

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


def _normalize_cjk_for_match(text: str, locale: str) -> str:
    """Applies light lossless normalization for CJK matching after raw DB lookup misses."""
    s = text.strip(" ,|\\'").replace('‧', '・').replace('·', '・').casefold()
    if locale == 'ja-jp':
        for dash in '‐-—―−':
            s = s.replace(dash, 'ー')
        s = _PROLONGED_STUTTER_RE.sub('ー', s)
    elif locale in ('zh-tw', 'zh-cn'):
        s = s.replace('‐', '-')
        if locale == 'zh-tw':
            s = s.replace('摇', '搖')
    return s


def _to_ko_jamo(text: str) -> str:
    """Decomposes Korean Hangul syllables into space-free compatibility jamo tokens."""
    nfd = unicodedata.normalize('NFD', text.replace(' ', ''))
    return ''.join(_HANGUL_JAMO_TO_COMPAT.get(ch, ch) for ch in nfd)


@functools.lru_cache(maxsize=None)
def _ja_kana_parts(ch: str) -> Tuple[str, str, bool, bool]:
    """Returns (large_hiragana_base, diacritic_marks, is_katakana, is_small) for a char."""
    is_small = ch in _JA_SMALL_CHARS
    is_kata = ('\u30a1' <= ch <= '\u30f6') or ch == 'ー'
    nfd = unicodedata.normalize('NFD', ch)
    marks = ''.join(c for c in nfd[1:] if c in ('\u3099', '\u309a'))
    base = nfd[0]
    if '\u30a1' <= base <= '\u30f6':
        base = chr(ord(base) - 0x60)
    base = base.translate(_JA_SMALL_TO_LARGE)
    return base, marks, is_kata, is_small


@functools.lru_cache(maxsize=None)
def _cjk_char_cost(a: str, b: str, locale: str) -> float:
    """Returns weighted substitution cost in [0.0, 1.0] for two CJK comparison tokens."""
    if a == b:
        return 0.0
    if locale == 'ko-kr':
        return 0.3 if b in _KO_JAMO_CONFUSIONS.get(a, ()) else 1.0
    if locale == 'ja-jp':
        if b in _JA_VISUAL_CONFUSIONS.get(a, ()):
            return 0.35
        ba, ma, ka, sa = _ja_kana_parts(a)
        bb, mb, kb, sb = _ja_kana_parts(b)
        if ba == bb:
            diffs = (ma != mb) + (ka != kb) + (sa != sb)
            if diffs == 1:
                return 0.2
            if diffs == 2:
                return 0.4
        return 1.0
    return 0.35 if b in _ZH_VISUAL_CONFUSIONS.get(a, ()) else 1.0


def _cjk_similarity(a: str, b: str, locale: str) -> float:
    """Computes normalized 0-100 similarity from confusion-weighted Levenshtein distance."""
    na, nb = len(a), len(b)
    max_len = max(na, nb)
    if max_len == 0:
        return 100.0
    prev = [float(j) for j in range(nb + 1)]
    for i, ca in enumerate(a, 1):
        curr = [float(i)]
        for j, cb in enumerate(b, 1):
            sub = prev[j - 1] + (0.0 if ca == cb else _cjk_char_cost(ca, cb, locale))
            curr.append(min(prev[j] + 1.0, curr[j - 1] + 1.0, sub))
        prev = curr
    return 100.0 * (1.0 - prev[nb] / max_len)


def _cjk_index_tokens(text: str, locale: str) -> Set[str]:
    """Extracts unigram and bigram keys used for fast CJK candidate generation."""
    if locale == 'ja-jp':
        text = ''.join(_ja_kana_parts(ch)[0] for ch in text)
    tokens = set(text)
    tokens.update(text[i:i + 2] for i in range(len(text) - 1))
    return tokens


@functools.lru_cache(maxsize=None)
def _get_cjk_search_index(locale: str) -> Tuple[
    Dict[str, str],
    List[Tuple[str, str]],
    Dict[str, Tuple[int, ...]],
    List[str],
]:
    """Builds and caches normalized lookup, representations, and char/bigram index for a CJK locale."""
    items = _get_item_list(locale)
    norm_db: Dict[str, str] = {}
    item_repr: List[Tuple[str, str]] = []
    raw_inv: Dict[str, List[int]] = collections.defaultdict(list)

    for idx, name in enumerate(items):
        norm_clean = _normalize_cjk_for_match(name, locale)
        norm_key = norm_clean.replace(' ', '') if locale == 'ko-kr' else norm_clean
        norm_db.setdefault(norm_key, name)
        rep = _to_ko_jamo(norm_clean) if locale == 'ko-kr' else norm_clean
        item_repr.append((name, rep))
        for tok in _cjk_index_tokens(rep, locale):
            raw_inv[tok].append(idx)

    inv_index = {tok: tuple(idxs) for tok, idxs in raw_inv.items()}
    rf_targets = [rep for _, rep in item_repr]
    return norm_db, item_repr, inv_index, rf_targets


def _match_cjk_item(norm_clean: str, locale: str) -> Optional[str]:
    """Runs Pass-2 confusion-aware matching for a CJK OCR string and returns best match if valid."""
    q_len = len(norm_clean)
    if q_len <= 2:
        return None

    _, item_repr, inv_index, rf_targets = _get_cjk_search_index(locale)
    cutoff, min_margin, high_conf = _fuzzy_thresholds(q_len)
    if q_len == 3:
        # Require low-cost confusion (score >= 78) on 3-char Pass-2 so unrelated 2/3 overlaps don't match.
        cutoff = max(cutoff, 78.0)

    q_rep = _to_ko_jamo(norm_clean) if locale == 'ko-kr' else norm_clean
    counts: collections.Counter[int] = collections.Counter()
    for tok in _cjk_index_tokens(q_rep, locale):
        for idx in inv_index.get(tok, ()):
            counts[idx] += 1

    cand_ids = {idx for idx, _ in counts.most_common(25)}
    for _, _, idx in process.extract(q_rep, rf_targets, scorer=fuzz.ratio, limit=15):
        cand_ids.add(idx)

    q_words = norm_clean.split() if locale == 'ko-kr' else []
    scored: List[Tuple[float, str]] = []
    for idx in cand_ids:
        cand_name, cand_rep = item_repr[idx]
        if locale == 'ko-kr' and len(q_words) > 1 and cand_name == q_words[-1]:
            continue
        scored.append((_cjk_similarity(q_rep, cand_rep, locale), cand_name))

    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_name = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    if best_score >= cutoff and (
        margin >= min_margin or (best_score >= high_conf and margin >= 1.5)
    ):
        return best_name
    if locale in ('zh-tw', 'zh-cn') and q_len >= 4 and best_score >= 58.0 and margin >= 12.0:
        return best_name
    return None


def match_items(item_names: Set[str], locale: str = 'en-us') -> Tuple[List[str], List[str]]:
    """Matches parsed OCR strings against the locale item database using exact + guarded fuzzy matching."""
    no_match_items: List[str] = []
    matched_items: Set[str] = set()
    item_db = _get_item_db(locale)
    clean_db = _get_clean_db_map(locale)
    item_list = _get_item_list(locale)
    aliases = LEGACY_ALIASES.get(locale, {})
    player_pattern = PLAYER_ITEM_PATTERNS.get(locale)
    is_cjk = locale in CJK_LOCALES
    cjk_norm_db = _get_cjk_search_index(locale)[0] if is_cjk else {}

    for raw_item in sorted(item_names):
        item = aliases.get(raw_item, raw_item)
        if item in item_db:
            matched_items.add(item)
            continue
        clean_key = item.replace('\u2060', '')
        if clean_key in clean_db:
            matched_items.add(clean_db[clean_key])
            continue

        if is_cjk:
            norm_clean = _normalize_cjk_for_match(item, locale)
            norm_key = norm_clean.replace(' ', '') if locale == 'ko-kr' else norm_clean
            if norm_key and norm_key in cjk_norm_db:
                matched_items.add(aliases.get(cjk_norm_db[norm_key], cjk_norm_db[norm_key]))
                continue
        else:
            norm_clean = item

        if player_pattern and player_pattern.match(item):
            logging.debug('Skipping player-named item %r', raw_item)
            continue

        cutoff, min_margin, high_conf = _fuzzy_thresholds(len(item))
        if cutoff < 100.0:
            candidates = process.extract(item, item_list, scorer=fuzz.ratio, limit=2, score_cutoff=cutoff)
            if candidates:
                best_name, best_score, _ = candidates[0]
                second_score = candidates[1][1] if len(candidates) > 1 else 0.0
                if best_score >= high_conf or (best_score - second_score) >= min_margin:
                    resolved = aliases.get(best_name, best_name)
                    logging.debug('Matched %r to %r (%.1f vs %.1f)', raw_item, resolved, best_score, second_score)
                    matched_items.add(resolved)
                    continue

        if locale == 'ru-eu' and len(item) >= 5:
            ru_items, ru_folded = _get_ru_folded_index()
            q_folded = item.translate(_RU_HOMOGLYPH_TRANS).casefold()
            ru_cands = process.extract(q_folded, ru_folded, scorer=fuzz.ratio, limit=2, score_cutoff=max(cutoff, 75.0))
            if ru_cands:
                _, best_score, best_idx = ru_cands[0]
                second_score = ru_cands[1][1] if len(ru_cands) > 1 else 0.0
                if best_score >= high_conf or (best_score - second_score) >= max(min_margin, 8.0):
                    best_ru = ru_items[best_idx]
                    resolved = aliases.get(best_ru, best_ru)
                    logging.debug('Matched Russian homoglyph %r to %r (%.1f)', raw_item, resolved, best_score)
                    matched_items.add(resolved)
                    continue

        if is_cjk:
            cjk_match = _match_cjk_item(norm_clean, locale)
            if cjk_match is not None:
                resolved = aliases.get(cjk_match, cjk_match)
                logging.debug('Matched CJK %r to %r', raw_item, resolved)
                matched_items.add(resolved)
                continue

        no_match_items.append(raw_item)

    if no_match_items and matched_items:
        filtered_unmatched: List[str] = []
        for u in no_match_items:
            if 2 <= len(u) <= 4 and ' ' not in u and any(m.startswith(u + ' ') for m in matched_items):
                logging.debug('Skipping partial-scroll prefix %r covered by matched item', u)
                continue
            filtered_unmatched.append(u)
        no_match_items = filtered_unmatched

    if no_match_items:
        logging.warning('Failed to match %d items: %s', len(no_match_items), no_match_items)
        # Allow a floor of 3 unmatched items on tiny catalogs so 1-3 OCR glitches don't abort them.
        assert len(no_match_items) <= max(3, int(0.3 * len(item_names))), \
            'Failed to match multiple items, wrong language?'

    return sorted(matched_items), no_match_items


def _read_frames(
    source: Union[str, cv2.VideoCapture],
    log_label: Optional[str] = None,
    step_state: Optional[List[int]] = None,
) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and returns the catalog list region in grayscale."""
    cap = cv2.VideoCapture(source) if isinstance(source, str) else source
    label = log_label or (source if isinstance(source, str) else '<video>')
    seen_valid = False
    pending_gap = 0
    total_gap_frames = 0
    valid_idx = 0
    scroll_positions: List[int] = []
    try:
        while True:
            if step_state is not None and seen_valid and pending_gap == 0 and not step_state[1]:
                base_step = step_state[0]
                grab_skip = (base_step - (valid_idx % base_step)) % base_step
            else:
                grab_skip = 0

            if grab_skip > 0:
                eof = False
                for _ in range(grab_skip):
                    if not cap.grab():
                        eof = True
                        break
                if eof:
                    break

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
                    pending_gap += grab_skip + 1
                continue

            gap_before = pending_gap
            if pending_gap:
                total_gap_frames += pending_gap
                pending_gap = 0
            else:
                valid_idx += grab_skip
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
                min_flips = max(5, 10 // max(1, step_state[0])) if step_state is not None else 10
                if _is_inconsistent_scroll(scroll_positions, min_flips=min_flips):
                    raise AssertionError('Video is scrolling inconsistently.')

            cur_valid_idx = valid_idx
            valid_idx += 1
            if step_state is not None and not step_state[1] and (cur_valid_idx % step_state[0]) != 0:
                continue

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
    col0 = frame[:, 0] < 200
    y_lines = list(col0.nonzero()[0])
    if not y_lines:
        return

    # Dashed divider lines are thin (1-3px) dark runs; ignore tall (~30px) hovered-row highlight blocks.
    thin_runs: List[float] = []
    run_start: Optional[int] = None
    for y in range(len(col0)):
        if col0[y] and run_start is None:
            run_start = y
        elif not col0[y] and run_start is not None:
            if y - run_start <= 5:
                thin_runs.append(0.5 * (run_start + y - 1))
            run_start = None
    if run_start is not None and len(col0) - run_start <= 5:
        thin_runs.append(0.5 * (run_start + len(col0) - 1))

    # Normalize row lines using circular mean modulo row height (53.45px) to avoid wrap-around jumps.
    pts = thin_runs if len(thin_runs) >= 3 else y_lines
    centroid = round(circular_phase(pts, 53.45))
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


def _is_inconsistent_scroll(scroll_positions: List[int], min_flips: int = 10) -> bool:
    """Detect when the user is not scrolling in a consistent direction (ignoring +-2px jitter)."""
    scroll_deltas = typing.cast(numpy.ndarray, numpy.diff(scroll_positions))
    downscroll_count = numpy.count_nonzero(scroll_deltas > 2)
    upscroll_count = numpy.count_nonzero(scroll_deltas < -2)
    return bool(downscroll_count > min_flips and upscroll_count > min_flips)


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
    with open(os.path.join(_BASE_DIR, 'items', f'{locale}.json'), encoding='utf-8') as fp:
        return set(json.load(fp))


@functools.lru_cache(maxsize=None)
def _get_clean_db_map(locale: str) -> Dict[str, str]:
    """Maps word-joiner-stripped item names to canonical item names."""
    return {it.replace('\u2060', ''): it for it in _get_item_db(locale)}


@functools.lru_cache(maxsize=None)
def _get_ru_folded_index() -> Tuple[List[str], List[str]]:
    """Returns (item_list, homoglyph-folded lowercase list) for Russian mixed-script matching."""
    items = _get_item_list('ru-eu')
    return items, [it.translate(_RU_HOMOGLYPH_TRANS).casefold() for it in items]


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

    nano_engine = _get_nano_ocr()
    if nano_engine is not None:
        try:
            return nano_engine.detect_locale(item_rows)
        except AssertionError:
            raise
        except Exception:
            logging.warning('Nano ONNX locale detection failed; falling back to Tesseract.', exc_info=True)

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
