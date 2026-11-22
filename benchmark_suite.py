"""Multi-locale and re-encode regression & performance benchmark suite (S4).

Runs real non-English (ja-jp, zh-tw, zh-cn, ko-kr, ru-eu, es-eu), English, and
re-encoded (.mov vs .mp4) catalog videos from `examples/more/`, verifying:
  1. 100% deterministic `locale='auto'` accuracy across all scripts/locales
  2. Stage-by-stage latency (`detect_mode`, `parse`, `detect_locale`, `ocr`, `match`, `total`)
  3. Exact DB hit rate, guarded fuzzy matches, unmatched items, and item counts
  4. `.mov` re-encode parity against `.mp4` (>= 99% item recall)

Usage:
  .venv\\Scripts\\python.exe benchmark_suite.py
  .venv\\Scripts\\python.exe benchmark_suite.py --save_baseline
"""

import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional

from absl import app
from absl import flags

import catalog
import scanner

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    'save_baseline',
    False,
    'If true, writes current metrics to testdata/multilocale_baseline.json when all cases pass.',
)
flags.DEFINE_string(
    'baseline_file',
    'testdata/multilocale_baseline.json',
    'Path to baseline JSON file for regression checks.',
)


@dataclasses.dataclass
class BenchmarkCase:
    name: str
    path: str
    expected_locale: str
    min_items: int


CASES: List[BenchmarkCase] = [
    BenchmarkCase('en_catalog', 'examples/catalog.mp4', 'en-us', 400),
    BenchmarkCase('en_twopage', 'examples/extra/catalog_twopage.mp4', 'en-us', 1200),
    BenchmarkCase('es_catalog', 'examples/more/catalog_es.mp4', 'es-eu', 350),
    BenchmarkCase('ru_catalog', 'examples/more/catalog_ru.mp4', 'ru-eu', 360),
    BenchmarkCase('ja_catalog_1', 'examples/more/catalog_jp.mp4', 'ja-jp', 230),
    BenchmarkCase('ja_catalog_2', 'examples/more/catalog_jp2.mp4', 'ja-jp', 340),
    BenchmarkCase('ja_catalog_3', 'examples/more/catalog_jp3.mp4', 'ja-jp', 740),
    BenchmarkCase('zhtw_catalog_1', 'examples/more/catalog_tw.mp4', 'zh-tw', 405),
    BenchmarkCase('zhtw_catalog_2', 'examples/more/catalog_tw2.mp4', 'zh-tw', 400),
    BenchmarkCase('zhcn_catalog_1', 'examples/more/catalog_cn.mp4', 'zh-cn', 750),
    BenchmarkCase('zhcn_catalog_2', 'examples/more/catalog_cn2.mp4', 'zh-cn', 160),
    BenchmarkCase('ko_catalog_1', 'examples/more/catalog_ko.mp4', 'ko-kr', 450),
    BenchmarkCase('ko_catalog_2', 'examples/more/catalog_ko2.mp4', 'ko-kr', 450),
    BenchmarkCase('en_ref_mp4', 'examples/more/acnh_catalog_ref.mp4', 'en-us', 1340),
    BenchmarkCase('en_reencoded_mov', 'examples/more/acnh_catalog_reencoded.mov', 'en-us', 1335),
]


def run_case(case: BenchmarkCase) -> Optional[Dict[str, Any]]:
    if not os.path.exists(case.path):
        return None

    try:
        t0 = time.perf_counter()
        mode = scanner._detect_media_type(case.path)
        assert mode == 'catalog', f'Expected catalog mode, got {mode!r}'
        t1 = time.perf_counter()
        rows = catalog.parse_video(case.path)
        t2 = time.perf_counter()
        detected_locale = catalog._detect_locale(rows, 'auto')
        t3 = time.perf_counter()
        raw_names = catalog.run_ocr(rows, lang=catalog.LOCALE_MAP[detected_locale])
        t4 = time.perf_counter()
        matched, unmatched = catalog.match_items(raw_names, detected_locale)
        t5 = time.perf_counter()
    except Exception as exc:
        return {'name': case.name, 'error': f'{type(exc).__name__}: {exc}'}

    db = catalog._get_item_db(detected_locale)
    aliases = catalog.LEGACY_ALIASES.get(detected_locale, {})
    exact_hits = sum(1 for n in raw_names if aliases.get(n, n) in db)
    exact_pct = (100.0 * exact_hits / len(raw_names)) if raw_names else 0.0

    return {
        'name': case.name,
        'mode': mode,
        'expected_locale': case.expected_locale,
        'detected_locale': detected_locale,
        'locale_ok': detected_locale == case.expected_locale,
        'rows': len(rows),
        'ocr_lines': len(raw_names),
        'exact_hits': exact_hits,
        'exact_pct': round(exact_pct, 1),
        'matched_items': len(matched),
        'unmatched_count': len(unmatched),
        'detect_mode_s': round(t1 - t0, 3),
        'parse_s': round(t2 - t1, 2),
        'detect_loc_s': round(t3 - t2, 2),
        'ocr_s': round(t4 - t3, 2),
        'match_s': round(t5 - t4, 3),
        'total_s': round(t5 - t0, 2),
        'items_set': set(matched),
        'min_items': case.min_items,
    }


def main(argv):
    del argv
    results: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []

    header = (
        f"{'Case':18s} | {'Locale':13s} | {'Rows':>4s} | {'Exact%':>6s} | "
        f"{'Items':>5s} | {'Unm':>3s} | {'Det':>5s} | {'Parse':>5s} | "
        f"{'Loc':>4s} | {'OCR':>5s} | {'Match':>5s} | {'Total':>6s}"
    )
    print(header)
    print('-' * len(header))

    t_suite = time.perf_counter()
    for case in CASES:
        res = run_case(case)
        if res is None:
            print(f'{case.name:18s} | SKIPPED (missing {case.path})')
            continue

        if 'error' in res:
            print(f"{case.name:18s} | ERROR: {res['error']}")
            failures.append(f"{case.name}: {res['error']}")
            continue

        results[case.name] = res
        loc_tag = f"{res['detected_locale']}" + ('' if res['locale_ok'] else f"(!={case.expected_locale})")
        print(
            f"{case.name:18s} | {loc_tag:13s} | {res['rows']:4d} | "
            f"{res['exact_pct']:5.1f}% | {res['matched_items']:5d} | {res['unmatched_count']:3d} | "
            f"{res['detect_mode_s']:4.2f}s | {res['parse_s']:4.2f}s | {res['detect_loc_s']:3.2f}s | "
            f"{res['ocr_s']:4.2f}s | {res['match_s']:4.2f}s | {res['total_s']:5.2f}s"
        )

        if not res['locale_ok']:
            failures.append(f"{case.name}: locale {res['detected_locale']} != {case.expected_locale}")
        if int(res['matched_items']) < case.min_items:
            failures.append(f"{case.name}: matched_items {res['matched_items']} < min_items {case.min_items}")

    total_suite_s = time.perf_counter() - t_suite
    print('-' * len(header))

    # Check .mov vs .mp4 parity if both ran.
    if 'en_ref_mp4' in results and 'en_reencoded_mov' in results:
        ref_items = results['en_ref_mp4']['items_set']
        mov_items = results['en_reencoded_mov']['items_set']
        overlap = len(ref_items & mov_items)
        recall = 100.0 * overlap / len(ref_items)
        print(
            f'.mov vs .mp4 parity: {len(mov_items)} vs {len(ref_items)} items '
            f'(recall={recall:.2f}%, missing={sorted(ref_items - mov_items)[:5]}, '
            f'extra={sorted(mov_items - ref_items)[:5]})'
        )
        if recall < 99.0:
            failures.append(f'.mov recall {recall:.2f}% < 99.0%')

    print(f'Completed {len(results)} benchmark cases in {total_suite_s:.2f}s')

    serializable = {
        k: {sk: sv for sk, sv in v.items() if sk != 'items_set'}
        for k, v in results.items()
    }

    if os.path.exists(FLAGS.baseline_file) and not FLAGS.save_baseline:
        with open(FLAGS.baseline_file, encoding='utf-8') as fp:
            baseline = json.load(fp)
        for name, cur in serializable.items():
            if name not in baseline:
                continue
            base_items = baseline[name]['matched_items']
            if cur['matched_items'] < base_items - 3:
                failures.append(
                    f"{name}: item count regressed from baseline {base_items} -> {cur['matched_items']}"
                )

    if FLAGS.save_baseline:
        if failures:
            print(f'Refusing to save baseline to {FLAGS.baseline_file} due to {len(failures)} failure(s).')
        else:
            with open(FLAGS.baseline_file, 'w', encoding='utf-8', newline='\n') as fp:
                json.dump(serializable, fp, indent=2)
                fp.write('\n')
            print(f'Saved baseline metrics to {FLAGS.baseline_file}')

    if failures:
        raise SystemExit('BENCHMARK FAILURES:\n  - ' + '\n  - '.join(failures))


if __name__ == '__main__':
    app.run(main)
