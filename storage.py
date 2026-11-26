from common import ScanMode, ScanResult, circular_phase, color_match, crop_720p_roi, slice_720p

import collections
import functools
import gzip
import json
import logging
import os

import cv2
import numpy

from typing import Any, Dict, Iterator, List, Optional, Tuple

# The expected color for the video background (BGR).
BG_COLOR = (69, 198, 246)

# The center color for an empty storage slot (BGR).
BLANK_COLOR = (192, 230, 242)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_STORAGE_ONNX_PATH = os.path.join(_BASE_DIR, 'models', 'acnh_storage_fp16.onnx')
_STORAGE_DB_PATH = os.path.join(_BASE_DIR, 'models', 'acnh_storage_db.npz')
_STORAGE_META_PATH = os.path.join(_BASE_DIR, 'models', 'acnh_storage_meta.json.gz')
_IMG_SIZE = 80
_BATCH_SIZE = 64
MIN_MATCH_SIM = 0.60


def _available_cpus() -> int:
    """Returns usable CPU count, respecting Linux cgroup/affinity limits when available."""
    if hasattr(os, 'sched_getaffinity'):
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    return os.cpu_count() or 2


class _StorageEmbedEngine:
    """Visual embedding matcher for ACNH 3.0 Home Storage icons + variations."""

    def __init__(self, onnx_path: str, db_path: str, meta_path: str) -> None:
        import onnxruntime as ort

        with gzip.open(meta_path, 'rb') as fp:
            meta = json.loads(fp.read().decode('utf-8'))
        self.names: List[str] = meta['names']
        self.rows: List[List[Any]] = meta['rows']
        self.locales: Dict[str, List[str]] = meta['locales']

        db_int8 = numpy.load(db_path)['embs']
        db_f32 = db_int8.astype(numpy.float32) * (1.0 / 127.0)
        norms = numpy.linalg.norm(db_f32, axis=-1, keepdims=True).clip(min=1e-6)
        self.db_embs: numpy.ndarray = numpy.ascontiguousarray(db_f32 / norms)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, _available_cpus())
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = True
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.sess.run(None, {'images': numpy.zeros((2, 3, _IMG_SIZE, _IMG_SIZE), dtype=numpy.float32)})

    def translate_name_id(self, name_id: int, locale: str) -> str:
        loc_table = self.locales.get(locale)
        if loc_table is not None:
            return loc_table[name_id]
        return self.names[name_id]

    def embed_tiles(self, item_images: List[numpy.ndarray]) -> numpy.ndarray:
        n = len(item_images)
        x = numpy.empty((n, 3, _IMG_SIZE, _IMG_SIZE), dtype=numpy.float32)
        for i, tile in enumerate(item_images):
            resized = cv2.resize(tile, (_IMG_SIZE, _IMG_SIZE), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x[i] = (rgb.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0) - 0.5) * 2.0

        out = numpy.empty((n, self.db_embs.shape[1]), dtype=numpy.float32)
        for start in range(0, n, _BATCH_SIZE):
            out[start:start + _BATCH_SIZE] = self.sess.run(None, {'images': x[start:start + _BATCH_SIZE]})[0]
        return out

    def match_icons(
        self,
        item_images: List[numpy.ndarray],
        locale: str = 'en-us',
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Matches storage tile crops against all 25.6k ACNH 3.0 icons, returning (details, unmatched)."""
        if not item_images:
            return [], []

        resolved_locale = 'en-us' if locale == 'auto' else locale
        embs = self.embed_tiles(item_images)
        sims = embs @ self.db_embs.T
        best_indices = sims.argmax(axis=-1)
        best_sims = sims[numpy.arange(len(item_images)), best_indices]

        counts: collections.Counter[Tuple[int, str, str]] = collections.Counter()
        unmatched: List[str] = []
        for idx, sim in zip(best_indices.tolist(), best_sims.tolist()):
            name_id, variant, pattern = self.rows[idx]
            if sim < MIN_MATCH_SIM:
                closest = self.translate_name_id(name_id, resolved_locale)
                logging.warning('Rejected storage tile with cosine similarity %.3f (closest: %s)', sim, closest)
                unmatched.append(f'{closest}?s={sim:.2f}')
                continue
            counts[(name_id, variant, pattern)] += 1

        details: List[Dict[str, Any]] = []
        for (name_id, variant, pattern), cnt in counts.items():
            details.append({
                'name': self.translate_name_id(name_id, resolved_locale),
                'variant': variant,
                'pattern': pattern,
                'count': cnt,
            })
        details.sort(key=lambda d: (d['name'], d['variant'], d['pattern']))
        return details, unmatched


@functools.lru_cache(maxsize=1)
def _get_storage_engine() -> _StorageEmbedEngine:
    return _StorageEmbedEngine(_STORAGE_ONNX_PATH, _STORAGE_DB_PATH, _STORAGE_META_PATH)


def detect(frame: numpy.ndarray) -> bool:
    """Detects if a given frame is showing the storage items."""
    color = slice_720p(frame, 0, 20, 1100, 1150).mean(axis=(0, 1))
    return color_match(color, BG_COLOR, 22)


def scan(video_file: str, locale: str = 'en-us') -> ScanResult:
    """Scans a video of scrolling through storage and returns all items found."""
    resolved_locale = 'en-us' if locale == 'auto' else locale
    item_images = parse_video(video_file)
    details, unmatched = match_items(item_images, locale=resolved_locale)
    items = sorted({d['name'] for d in details})

    return ScanResult(
        mode=ScanMode.STORAGE,
        items=items,
        locale=resolved_locale,
        unmatched=unmatched,
        details=details,
    )


def parse_video(filename: str) -> List[numpy.ndarray]:
    """Parses a whole video and returns images for all storage items found."""
    all_rows: List[numpy.ndarray] = []
    for frame in _read_frames(filename):
        for new_row in _parse_frame(frame):
            if _is_duplicate_row(all_rows, new_row):
                continue  # Skip non-moving frames
            all_rows.extend(new_row)
    icons = _remove_blanks(all_rows)
    assert icons, 'No storage items found, invalid video?'
    return icons


def match_items(
    item_images: List[numpy.ndarray],
    locale: str = 'en-us',
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Matches icons against the ACNH 3.0 storage icon database, returning (details, unmatched)."""
    return _get_storage_engine().match_icons(item_images, locale=locale)


def translate_names(item_names: List[str], locale: str) -> List[str]:
    """Translates a list of canonical English storage item names to the given locale."""
    if locale in ('en-us', 'auto'):
        return item_names
    engine = _get_storage_engine()
    loc_table = engine.locales.get(locale)
    if loc_table is None:
        return item_names
    name_to_id = {n: i for i, n in enumerate(engine.names)}
    return [loc_table[name_to_id[n]] if n in name_to_id else n for n in item_names]


def _read_frames(filename: str) -> Iterator[numpy.ndarray]:
    """Parses frames of the given video and yields the 720p-normalized storage grid region."""
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
            # Crop the region containing storage items at 720p-normalized coordinates.
            yield crop_720p_roi(frame, 150, 675, 112, 1168)
    finally:
        cap.release()


def _parse_frame(frame: numpy.ndarray) -> Iterator[List[numpy.ndarray]]:
    """Parses an individual frame and extracts 100x100 item tiles from the storage grid."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x_positions = list(range(0, 1056, 132))

    cols = []
    for x in x_positions[1:]:
        empty_col = gray[:, x - 10:x + 10]
        if empty_col.min() < 200:
            continue  # Skip columns with occlusions
        cols.append(empty_col)

    if not cols:
        return

    thresh = cv2.threshold(cv2.hconcat(cols), 236, 255, 0)[1]
    separators = list(numpy.nonzero(thresh.mean(axis=1) < 240)[0])
    if not separators:
        return

    # Normalize row lines using circular mean modulo row height (127px).
    centroid = round(circular_phase(separators, 127.0))
    y_positions = list(range(centroid, 525, 127))

    for y in y_positions:
        if y + 127 > frame.shape[0]:
            continue  # Past the bottom of the frame.
        # Skip row when teal tooltip overlaps the inter-row gap below the tile crop.
        tooltip = cv2.inRange(frame[y + 116:y + 127, :], (135, 160, 45), (205, 235, 120))
        if tooltip.mean() > 6:
            continue

        yield [frame[y + 13:y + 113, x + 16:x + 116] for x in x_positions]


def _is_duplicate_row(all_rows: List[numpy.ndarray], new_row: List[numpy.ndarray]) -> bool:
    """Checks if the new row is the same as the previously seen rows."""
    if not new_row or len(all_rows) < len(new_row):
        return False

    new_concat = cv2.hconcat(new_row)
    # Checks the last 4 rows for similarities to the newly added row.
    for ind in [slice(-8, None), slice(-16, -8), slice(-24, -16), slice(-32, -24)]:
        old_slice = all_rows[ind]
        if len(old_slice) != len(new_row):
            continue
        old_concat = cv2.hconcat(old_slice)
        if cv2.absdiff(new_concat, old_concat).mean() < 12:
            # If the old version had a cursor in it, replace with new row.
            if old_concat[-5:].min() < 50:
                all_rows[ind] = new_row
            return True

    return False


def _remove_blanks(all_icons: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """Removes all tiles that show empty storage slots."""
    filtered_icons = []
    for icon in all_icons:
        center_color = icon[40:60, 40:60].mean(axis=(0, 1))
        if numpy.linalg.norm(center_color - BLANK_COLOR) < 12:
            continue
        filtered_icons.append(icon)
    return filtered_icons


if __name__ == '__main__':
    results = scan('examples/extra/storage_sample.mp4')
    print('\n'.join(results.items))
