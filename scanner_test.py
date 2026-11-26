import json
import os
import shutil
import tempfile
from unittest import mock

import cv2
import numpy
from absl.testing import absltest, parameterized

import catalog
import critters
import music
import reactions
import recipes
import scanner
import storage
from common import ScanMode, crop_720p_roi, is_image_file, slice_720p


def _fetch_ground_truth():
    with open('testdata/examples.json', encoding='utf-8') as fp:
        yield json.load(fp)
    with open('testdata/extra.json', encoding='utf-8') as fp:
        yield json.load(fp)
    with open('testdata/locale_cases.json', encoding='utf-8') as fp:
        yield json.load(fp)


GROUND_TRUTH, EXTRA_GROUND_TRUTH, LOCALE_CASES = _fetch_ground_truth()


class ScannerTest(absltest.TestCase):

    def setUp(self):
        self.maxDiff = None
        return super().setUp()

    @property
    def name(self):
        return self.id().split('.')[-1]

    def test_catalog(self):
        results = scanner.scan_media('examples/catalog.mp4')
        self.assertEqual(results.mode, ScanMode.CATALOG)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_catalog_forsale(self):
        results = scanner.scan_media('examples/catalog.mp4', for_sale=True)
        self.assertEqual(results.mode, ScanMode.CATALOG)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_recipes(self):
        results = scanner.scan_media('examples/recipes.mp4')
        self.assertEqual(results.mode, ScanMode.RECIPES)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_recipes_translate(self):
        results = scanner.scan_media('examples/recipes.mp4', locale='fr-eu')
        self.assertEqual(results.mode, ScanMode.RECIPES)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'fr-eu')

    def test_critters(self):
        results = scanner.scan_media('examples/critters.mp4')
        self.assertEqual(results.mode, ScanMode.CRITTERS)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_critters_translate(self):
        results = scanner.scan_media('examples/critters.mp4', locale='ko-kr')
        self.assertEqual(results.mode, ScanMode.CRITTERS)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'ko-kr')

    def test_reactions(self):
        results = scanner.scan_media('examples/reactions.jpg')
        self.assertEqual(results.mode, ScanMode.REACTIONS)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_reactions_translate(self):
        results = scanner.scan_media('examples/reactions.jpg', locale='de-eu')
        self.assertEqual(results.mode, ScanMode.REACTIONS)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'de-eu')

    def test_music(self):
        results = scanner.scan_media('examples/music.mp4')
        self.assertEqual(results.mode, ScanMode.MUSIC)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_music_translate(self):
        results = scanner.scan_media('examples/music.mp4', locale='ja-jp')
        self.assertEqual(results.mode, ScanMode.MUSIC)
        self.assertEqual(results.items, GROUND_TRUTH[self.name])
        self.assertEqual(results.locale, 'ja-jp')

    def test_legacy_aliases_and_fuzzy_guards(self):
        matched, unmatched = catalog.match_items(
            {
                'Writing desk', 'Teacup ride', 'Fish and chips', 'Tee with silicon bib',
                'Beige desert-tile flooring', 'Blue desert-tile flooring',
                'Olive desert-tile wall', 'Purple desert-tile flooring', 'Paper bag',
            },
            locale='en-us',
        )
        self.assertEqual(
            matched,
            [
                'Beige Moroccan flooring', 'Blue Moroccan flooring', 'Fish-and-chips',
                'Olive Moroccan wall', 'Paper-bag hood', 'Plaza teacup ride',
                'Purple Moroccan flooring', 'Study desk', 'Tee with silicone bib',
            ],
        )
        self.assertEqual(unmatched, [])

        # Short noisy CJK strings must not false-match unrelated items.
        ja_matched, ja_unmatched = catalog.match_items(
            {'アーガイルベスト', 'あきいろチェックワンピース', 'あさがおのゆかた', 'アノラック', 'ごの'},
            locale='ja-jp',
        )
        self.assertIn('ごの', ja_unmatched)
        self.assertNotIn('みの', ja_matched)

        # Verify items/legacy_aliases.json integrity across all 14 locales + non-English renames.
        for loc in catalog.LOCALE_TIEBREAK_ORDER:
            db = catalog._get_item_db(loc)
            for old_name, new_name in catalog.LEGACY_ALIASES.get(loc, {}).items():
                self.assertNotIn(old_name, db, f'{loc} alias key {old_name!r} already in DB')
                self.assertIn(new_name, db, f'{loc} alias target {new_name!r} missing from DB')

        es_matched, es_unmatched = catalog.match_items({'Máquina de coser antigua'}, locale='es-eu')
        self.assertEqual(es_matched, ['Máquina de coser añeja'])
        self.assertEqual(es_unmatched, [])

        # Russian legacy renames + Cyrillic/Latin homoglyph folding for mixed-script tees.
        ru_matched, ru_unmatched = catalog.match_items(
            {'Кусочки пластыря', 'Миленькое платье', 'Футболка Аппуеоng'},
            locale='ru-eu',
        )
        self.assertEqual(ru_matched, ['Пластырь', 'Платье из грез', 'Футболка Annyeong'])
        self.assertEqual(ru_unmatched, [])

    def test_cjk_confusion_matching(self):
        # Note: keep total noisy items <= max(3, int(0.3 * len(items))) so assert isn't tripped.
        ja_matched, ja_unmatched = catalog.match_items(
            {
                'アーガイルベスト', 'あきいろチェックワンピース', 'あさがおのゆかた',
                'アノラック', 'あみ', 'おり', 'かめ', 'そり',
                'ポシテージ', 'スワック,', '避三ールチェア', 'エジェタターラツタ',
                'フルーツウォーーターサーバー', 'ボンテ一ジ',
                'ごの', 'アップ', 'ジック',
            },
            locale='ja-jp',
        )
        self.assertEqual(
            set(ja_matched),
            {
                'アーガイルベスト', 'あきいろチェックワンピース', 'あさがおのゆかた',
                'アノラック', 'あみ', 'おり', 'かめ', 'そり',
                'エフェクターラック', 'スワッグ', 'ビニールチェア',
                'フルーツウォーターサーバー', 'ボンテージ',
            },
        )
        self.assertEqual(ja_unmatched, ['ごの', 'アップ', 'ジック'])

        ko_matched, ko_unmatched = catalog.match_items(
            {
                '1UP버섯', '3D안경', '5층탑', '거미', '검', '고민', '곤란', '광차',
                '짐볼', '닷 장식', '발랫대마', '70 피규어', '7ᅮ0 오디오',
                '빙글빙글커피잔', '헬로 티셔츠', '헬로',
                '불', '분추', '이스터 풍선 6',
            },
            locale='ko-kr',
        )
        for expected in ('TOY 오디오', 'TOY 피규어', '광장의 커피 잔', '닻 장식', '빨랫대야', '짐 볼', '헬로 티셔츠'):
            self.assertIn(expected, ko_matched)
        self.assertEqual(ko_unmatched, ['분추', '불', '이스터 풍선 6'])

        tw_matched, tw_unmatched = catalog.match_items(
            {
                '冰桌', '冰桶', '冰箱', '和服架', '飛鏢靶',
                '樸春DIY作業台', '穴烤爐', '波斯菊花決',
                'toy床鋪', '可愛diy桌', '搬家紙箱m', '摇椅', '中性色轉角',
                '復活節廟子', '火火',
            },
            locale='zh-tw',
        )
        for expected in (
            'TOY床鋪', '中立轉角', '可愛DIY桌', '搬家紙箱M',
            '搖椅', '樸素DIY作業台', '波斯菊花灑', '窯烤爐',
        ):
            self.assertIn(expected, tw_matched)
        self.assertEqual(tw_unmatched, ['復活節廟子', '火火'])

        cn_matched, cn_unmatched = catalog.match_items(
            {
                '书堆', '冰桶', '户外长凳', '电子闹钟', '黄金柏',
                '青蔡庭院五', '水果蓝', '公园攀仆加', '旋转咖啡杯',
                '复活节受子',
            },
            locale='zh-cn',
        )
        for expected in ('公园攀爬架', '广场咖啡杯', '水果篮', '青苔庭院石'):
            self.assertIn(expected, cn_matched)
        self.assertEqual(cn_unmatched, ['复活节受子'])

    def test_nano_onnx_ocr_and_fallback(self):
        row = numpy.full((35, 415), 245, dtype=numpy.uint8)
        row[8:26, 20:110] = 45
        prepped, ink_ws = catalog._NanoOcrEngine.prep_rows([row])
        self.assertEqual(prepped.shape, (1, 32, 384))
        self.assertLess(int(ink_ws[0]), 384)

        rows = catalog.parse_video('examples/extra/catalog_zhcn.mp4')[:6]
        det_loc = catalog._detect_locale(rows, 'auto')
        self.assertEqual(det_loc, 'zh-cn')
        self.assertIsNotNone(getattr(catalog._OCR_TLS, 'sample_cache', None))
        ocr_names = catalog.run_ocr(rows, lang='chi_sim', locale='zh-cn')
        self.assertIsNone(getattr(catalog._OCR_TLS, 'sample_cache', None))
        db = catalog._get_item_db('zh-cn')
        self.assertGreaterEqual(sum(1 for name in ocr_names if name in db), 5)

        with mock.patch.object(catalog, '_get_nano_ocr', return_value=None):
            with mock.patch.object(catalog, '_ocr_chunk', return_value={'テスト'}) as mock_tess:
                self.assertEqual(catalog.run_ocr([row], lang='jpn'), {'テスト'})
                mock_tess.assert_called_once()

        failing_eng = mock.MagicMock()
        failing_eng.recognize_rows.side_effect = RuntimeError('boom')
        with mock.patch.object(catalog, '_get_nano_ocr', return_value=failing_eng):
            with mock.patch.object(catalog, '_ocr_chunk', return_value={'테스트'}) as mock_tess:
                self.assertEqual(catalog.run_ocr([row], lang='kor'), {'테스트'})
                mock_tess.assert_called_once()

    def test_player_named_items_skipped(self):
        matched, unmatched = catalog.match_items(
            {"Ehsan's tent", "Ehsan's kit", "Bob's tent", "Kids' tent"},
            locale='en-us',
        )
        self.assertEqual(matched, ["Kids' tent"])
        self.assertEqual(unmatched, [])

        fr_matched, fr_unmatched = catalog.match_items(
            {'Tente de Ehsan', 'Kit de chimie'},
            locale='fr-eu',
        )
        self.assertEqual(fr_matched, ['Kit de chimie'])
        self.assertEqual(fr_unmatched, [])

        ja_matched, ja_unmatched = catalog.match_items(
            {'えさんのテント', 'アーガイルベスト'},
            locale='ja-jp',
        )
        self.assertEqual(ja_matched, ['アーガイルベスト'])
        self.assertEqual(ja_unmatched, [])

        ko_matched, ko_unmatched = catalog.match_items(
            {'에산의 텐트', '에산 이전 키트', '박물관 이전 키트'},
            locale='ko-kr',
        )
        self.assertEqual(ko_matched, ['박물관 이전 키트'])
        self.assertEqual(ko_unmatched, [])

    def test_cleanup_name_whole_word_bail(self):
        self.assertEqual(catalog._cleanup_name('Moss bail', 'eng'), 'Moss ball')
        self.assertEqual(catalog._cleanup_name('Bail', 'eng'), 'Ball')
        self.assertEqual(catalog._cleanup_name('Máscara de baile', 'spa'), 'Máscara de baile')
        self.assertEqual(catalog._cleanup_name('Bailarina Trini', 'spa'), 'Bailarina Trini')

    def test_catalog_normalize_hovered_row(self):
        normal_row = numpy.full((35, 415), 245, dtype=numpy.uint8)
        normal_row[8:26, 20:100] = 55
        out_normal, is_hov = catalog._normalize_row(normal_row)
        self.assertFalse(is_hov)
        self.assertTrue(numpy.array_equal(out_normal, normal_row))

        hovered_row = numpy.full((35, 415), 208, dtype=numpy.uint8)
        hovered_row[:, ::2] = 220
        hovered_row[8:26, 20:100] = 35
        out_hov, is_hov2 = catalog._normalize_row(hovered_row)
        self.assertTrue(is_hov2)
        self.assertGreaterEqual(int(out_hov[:, 350:415].min()), 240)
        self.assertEqual(int(out_hov[8:26, 20:100].min()), 35)

    def test_catalog_align_diff_distinguishes_single_char(self):
        row_a = numpy.full((35, 415), 245, dtype=numpy.uint8)
        row_a[10:26, 16:196] = 60
        row_shifted = numpy.full_like(row_a, 245)
        row_shifted[1:, :] = row_a[:-1, :]
        mean_same, blk_same, dy_same = catalog._align_row_diff(row_a, row_shifted, 200)
        self.assertEqual(dy_same, 1)
        self.assertLess(mean_same, 2.0)
        self.assertLess(blk_same, 5.0)

        row_b = row_a.copy()
        row_b[12:24, 96:112] = 180  # Differ in one 16px character block inside a longer label
        mean_diff, blk_diff, _ = catalog._align_row_diff(row_a, row_b, 200)
        self.assertLess(mean_diff, catalog.ROW_DEDUPE_MEAN_THRESH)
        self.assertGreater(blk_diff, catalog.ROW_DEDUPE_BLOCK_THRESH)

    def test_case_insensitive_image_extensions(self):
        self.assertTrue(is_image_file('photo.JPG'))
        self.assertTrue(is_image_file('page_%d.PNG'))
        self.assertFalse(is_image_file('clip.MP4'))

        with tempfile.TemporaryDirectory() as tmpdir:
            upper_jpg = os.path.join(tmpdir, 'reactions_upper.JPG')
            shutil.copyfile('examples/reactions.jpg', upper_jpg)
            results = scanner.scan_media(upper_jpg)
            self.assertEqual(results.items, GROUND_TRUTH['test_reactions'])

            critter_upper = os.path.join(tmpdir, 'critter_sea.JPG')
            shutil.copyfile('examples/extra/critters_seacreatures_img.jpg', critter_upper)
            c_res = scanner.scan_media(critter_upper)
            self.assertEqual(c_res.items, EXTRA_GROUND_TRUTH['critters_seacreatures_img.jpg'])

    def test_slice_and_crop_720p_roi(self):
        frame_720 = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
        sliced_720 = slice_720p(frame_720, 150, 160, -20, None)
        self.assertEqual(sliced_720.shape, (10, 20, 3))
        self.assertTrue(numpy.shares_memory(sliced_720, frame_720))

        cropped_720 = crop_720p_roi(frame_720, 110, 720, 45, 730)
        self.assertEqual(cropped_720.shape, (610, 685, 3))
        self.assertTrue(numpy.shares_memory(cropped_720, frame_720))

        frame_1080 = numpy.zeros((1080, 1920, 3), dtype=numpy.uint8)
        sliced_1080 = slice_720p(frame_1080, None, 20, -20, None)
        self.assertEqual(sliced_1080.shape, (30, 30, 3))

        cropped_1080 = crop_720p_roi(frame_1080, 110, 720, 45, 730)
        self.assertEqual(cropped_1080.shape, (610, 685, 3))
        self.assertEqual(crop_720p_roi(frame_1080, 149, 623, None, None).shape, (474, 1280, 3))
        self.assertEqual(crop_720p_roi(frame_1080, 70, 75, 65, 260).shape, (5, 195, 3))

    def test_music_phash_rejects_noise_cover(self):
        rng = numpy.random.default_rng(42)
        noise_cover = rng.integers(0, 256, size=(260, 260, 3), dtype=numpy.uint8)
        matched, unmatched = music.match_songs([noise_cover])
        self.assertEqual(matched, [])
        self.assertEqual(len(unmatched), 1)
        self.assertIn('?d=', unmatched[0])

    def test_icon_matchers_reject_noise(self):
        rng = numpy.random.default_rng(42)
        noise_recipe = rng.integers(0, 256, size=(112, 112, 3), dtype=numpy.uint8)
        r_matched, r_unmatched = recipes.match_recipes([noise_recipe])
        self.assertEqual(r_matched, [])
        self.assertEqual(len(r_unmatched), 1)
        self.assertIn('?d=', r_unmatched[0])

        noise_critter = rng.integers(0, 256, size=(80, 80, 3), dtype=numpy.uint8).view(critters.CritterIcon)
        noise_critter.critter_type = critters.CritterType.INSECTS
        c_matched, c_unmatched = critters.match_critters([noise_critter])
        self.assertEqual(c_matched, [])
        self.assertEqual(len(c_unmatched), 1)
        self.assertIn('?d=', c_unmatched[0])

        noise_reaction = rng.integers(0, 256, size=(64, 64, 3), dtype=numpy.uint8)
        re_matched, re_unmatched = reactions.match_reactions([noise_reaction])
        self.assertEqual(re_matched, [])
        self.assertEqual(len(re_unmatched), 1)
        self.assertIn('?d=', re_unmatched[0])

    def test_rejects_low_resolution_media(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for width, height in ((640, 360), (426, 240)):
                path = os.path.join(tmpdir, f'low_{width}x{height}.png')
                cv2.imwrite(path, numpy.zeros((height, width, 3), dtype=numpy.uint8))
                with self.assertRaisesRegex(AssertionError, f'Invalid resolution: {width}x{height}'):
                    scanner.scan_media(path)

    def test_accepts_480p_media(self):
        img = cv2.imread('examples/reactions.jpg')
        self.assertIsNotNone(img)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_480 = os.path.join(tmpdir, 'reactions_480.jpg')
            cv2.imwrite(img_480, cv2.resize(img, (854, 480), interpolation=cv2.INTER_AREA))
            r_res = scanner.scan_media(img_480)
            self.assertEqual(r_res.mode, ScanMode.REACTIONS)
            self.assertEqual(r_res.items, GROUND_TRUTH['test_reactions'])
            self.assertEqual(r_res.unmatched, [])

            vid_480 = os.path.join(tmpdir, 'catalog_tiny_480.mp4')
            cap = cv2.VideoCapture('examples/extra/catalog_tiny.mp4')
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(vid_480, cv2.VideoWriter_fourcc(*'mp4v'), fps, (854, 480))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(cv2.resize(frame, (854, 480), interpolation=cv2.INTER_AREA))
            cap.release()
            writer.release()

            c_res = scanner.scan_media(vid_480, locale='auto')
            self.assertEqual(c_res.mode, ScanMode.CATALOG)
            self.assertEqual(c_res.locale, 'en-us')
            self.assertEqual(len(c_res.items), 13)
            self.assertEqual(c_res.unmatched, [])

    def test_rejects_over_duration_video(self):
        mock_cap = mock.MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 6000.0,
        }.get(prop, 0.0)
        with mock.patch.object(cv2, 'VideoCapture', return_value=mock_cap):
            with self.assertRaisesRegex(AssertionError, 'Your video is too long.'):
                scanner.scan_media('fake_long.mp4')
        mock_cap.release.assert_called_once()

    def test_rejects_corrupt_and_non_acnh_media(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_vid = os.path.join(tmpdir, 'empty.mp4')
            with open(empty_vid, 'wb'):
                pass
            with self.assertRaisesRegex(AssertionError, 'Media is not showing a known scan type.'):
                scanner.scan_media(empty_vid)

            blank_img = os.path.join(tmpdir, 'non_acnh.png')
            cv2.imwrite(blank_img, numpy.full((720, 1280, 3), 40, dtype=numpy.uint8))
            with self.assertRaisesRegex(AssertionError, 'Media is not showing a known scan type.'):
                scanner.scan_media(blank_img)

    def test_webp_and_png_image_inputs(self):
        img = cv2.imread('examples/reactions.jpg')
        self.assertIsNotNone(img)
        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in ('.webp', '.png'):
                out_path = os.path.join(tmpdir, f'reactions{ext}')
                self.assertTrue(cv2.imwrite(out_path, img))
                results = scanner.scan_media(out_path)
                self.assertEqual(results.mode, ScanMode.REACTIONS)
                self.assertEqual(results.items, GROUND_TRUTH['test_reactions'])
                self.assertEqual(results.unmatched, [])

    def test_explicit_locale_override(self):
        results = scanner.scan_media('examples/extra/catalog_tiny.mp4', locale='en-us')
        self.assertEqual(results.mode, ScanMode.CATALOG)
        self.assertEqual(results.locale, 'en-us')
        for required_item in LOCALE_CASES['catalog_tiny.mp4']['required_items']:
            self.assertIn(required_item, results.items)
        self.assertEqual(len(results.items), 13)
        self.assertEqual(results.unmatched, [])

    def test_all_14_locales_nano_vocab_and_matching(self):
        engine = catalog._get_nano_ocr()
        self.assertIsNotNone(engine)
        assert engine is not None
        vocab_chars = set(engine.vocab[1:])

        for loc in catalog.LOCALE_TIEBREAK_ORDER:
            self.assertIn(loc, engine.allowed_by_locale)
            db = catalog._get_item_db(loc)
            aliases = catalog.LEGACY_ALIASES.get(loc, {})
            missing = set(''.join(db)) - vocab_chars
            self.assertEqual(missing, set(), f'{loc} DB has chars missing from Nano vocab: {missing}')

            # Spot-check exact match, legacy alias, and word-joiner (\u2060) stripping per locale.
            sample_item = next(it for it in sorted(db) if len(it) >= 4)
            probes = {sample_item}
            if aliases:
                alias_src, alias_dst = next(iter(aliases.items()))
                probes.add(alias_src)
            else:
                alias_dst = None

            wj_items = [it for it in db if '\u2060' in it]
            if wj_items:
                probes.add(wj_items[0].replace('\u2060', ''))

            matched, unmatched = catalog.match_items(probes, locale=loc)
            self.assertEqual(unmatched, [], f'{loc} failed matching probes: {unmatched}')
            self.assertIn(sample_item, matched)
            if alias_dst:
                self.assertIn(alias_dst, matched)
            if wj_items:
                self.assertIn(wj_items[0], matched)

    def test_nano_detect_locale_fuzzy_fallback_and_errors(self):
        engine = catalog._get_nano_ocr()
        self.assertIsNotNone(engine)
        assert engine is not None
        rows = [numpy.full((35, 415), 245, dtype=numpy.uint8) for _ in range(4)]

        # When exact hits are 0 across all locales, fuzzy fallback should pick French cleanly.
        fr_typos = ['Tablier de bricolagee', 'Établi de bricolagee', 'Baignoire en bois de cypres']
        with mock.patch.object(engine, 'decode_masked', return_value=fr_typos):
            det_loc = engine.detect_locale(rows)
        self.assertEqual(det_loc, 'fr-eu')
        catalog._OCR_TLS.sample_cache = None

        # Pure garbage with zero exact and zero fuzzy hits must fail loudly.
        with mock.patch.object(engine, 'decode_masked', return_value=['zzzzqqqq', 'xxxxwwww']):
            with self.assertRaisesRegex(AssertionError, 'Failed to automatically detect language.'):
                engine.detect_locale(rows)
        catalog._OCR_TLS.sample_cache = None

    def test_catalog_edge_cases_cleanup_for_sale_and_thresholds(self):
        self.assertEqual(catalog._cleanup_name('’Ao dai‘', 'eng'), "'Áo dài'")
        self.assertEqual(catalog._cleanup_name('Моок пс. тс.', 'rus'), 'Nook Inc. Inc.')

        # Too many unmatched items (> max(3, 30%)) must raise.
        with self.assertRaisesRegex(AssertionError, 'Failed to match multiple items'):
            catalog.match_items({'zzz_bad_1', 'zzz_bad_2', 'zzz_bad_3', 'zzz_bad_4'}, locale='en-us')

        # for_sale=True should filter out rows whose price area is blank/light (>100).
        frame = numpy.full((480, 585), 245, dtype=numpy.uint8)
        frame[100, 0] = 100
        frame[153, 0] = 100
        # Row 1 (y=100 -> 60:95): item text + dark price indicator
        frame[65:90, 20:150] = 40
        frame[70:85, 440:470] = 40
        # Row 2 (y=153 -> 113:148): item text only (not for sale)
        frame[118:143, 20:150] = 40
        all_rows = list(catalog._parse_frame(frame, for_sale=False))
        sale_rows = list(catalog._parse_frame(frame, for_sale=True))
        self.assertGreater(len(all_rows), 1)
        self.assertEqual(len(sale_rows), 1)

    def test_parse_video_cap_grab_parity(self):
        # Verify cap.grab() frame skipping yields the exact same row count as reading all frames.
        rows_grab = catalog.parse_video('examples/extra/catalog_twopage.mp4')
        self.assertEqual(len(rows_grab), 1320)

    def test_storage_scan_and_details(self):
        res = scanner.scan_media('examples/extra/storage_sample.mp4')
        self.assertEqual(res.mode, ScanMode.STORAGE)
        self.assertEqual(res.locale, 'en-us')
        self.assertEqual(res.unmatched, [])
        self.assertEqual(len(res.items), 86)
        self.assertIsNotNone(res.details)
        assert res.details is not None
        self.assertEqual(len(res.details), 88)
        self.assertEqual(sum(d['count'] for d in res.details), 98)
        self.assertIn({'name': 'Bath-towel wrap', 'variant': 'Green', 'pattern': '', 'count': 1}, res.details)

        res_ja = scanner.scan_media('examples/extra/storage_sample.mp4', locale='ja-jp')
        self.assertEqual(res_ja.locale, 'ja-jp')
        self.assertEqual(len(res_ja.items), 86)
        self.assertIn('TOYなフィギュア', res_ja.items)

    def test_storage_rejects_noise(self):
        rng = numpy.random.default_rng(42)
        noise_tile = rng.integers(0, 256, size=(100, 100, 3), dtype=numpy.uint8)
        details, unmatched = storage.match_items([noise_tile])
        self.assertEqual(details, [])
        self.assertEqual(len(unmatched), 1)
        self.assertIn('?s=', unmatched[0])

    def test_catalog_parse_frame_ignores_hover_bleed_and_wraparound(self):
        # Divider lines placed near the 0 / 53.45 wrap-around boundary + a 30px hover bar in col 0.
        frame = numpy.full((480, 585), 245, dtype=numpy.uint8)
        for y in (53, 107, 160, 214, 267, 321, 374, 428):
            frame[y:y + 2, 0] = 100
        # Simulated dark hovered-row block in column 0 between dividers (y=115..145).
        frame[115:145, 0] = 160
        # Place distinct ink at y=160-25=135 (centered in row 160 -> 120:155)
        frame[130:142, 20:120] = 30
        rows = list(catalog._parse_frame(frame, for_sale=False))
        self.assertEqual(len(rows), 8)
        # Row corresponding to y=160 (index 2: 53, 107, 160) should contain the ink cleanly centered.
        self.assertLess(int(rows[2][10:22, 20:120].mean()), 60)

    def test_crop_black_bars_and_dimmed_color_match(self):
        import common
        # 1) Normal 1280x720 is untouched.
        img_720 = numpy.full((720, 1280, 3), 180, dtype=numpy.uint8)
        self.assertEqual(common.crop_black_bars(img_720).shape, (720, 1280, 3))

        # 2) Pillarboxed 2796x1290 (iPhone landscape) crops to 16:9 (~2293x1290).
        pillar = numpy.zeros((1290, 2796, 3), dtype=numpy.uint8)
        pillar[:, 252:2545] = (180, 253, 254)
        pillar[:36, 252:2545] = (110, 233, 238)
        cropped = common.crop_black_bars(pillar)
        self.assertEqual(cropped.shape[:2], (1290, 2293))
        self.assertTrue(catalog.detect(pillar))

        # 3) Letterboxed 1179x2556 (iPhone portrait) crops to 16:9 (1179x663).
        letter = numpy.zeros((2556, 1179, 3), dtype=numpy.uint8)
        letter[946:1609, :] = (254, 221, 244)
        cropped_l = common.crop_black_bars(letter)
        self.assertEqual(cropped_l.shape[:2], (663, 1179))
        self.assertTrue(reactions.detect(letter))

        # 4) Uniformly dimmed (0.90x) catalog & recipes backgrounds still detect, off-hue rejects.
        dim_cat = numpy.full((720, 1280, 3), (162, 230, 236), dtype=numpy.uint8)
        dim_cat[:20, -20:] = (96, 206, 219)
        self.assertTrue(catalog.detect(dim_cat))
        dim_rec = numpy.full((720, 1280, 3), (174, 201, 209), dtype=numpy.uint8)
        self.assertTrue(recipes.detect(dim_rec))
        self.assertFalse(common.color_match((180, 180, 250), catalog.SIDE_COLOR, 22))

    def test_storage_skips_compressed_teal_tooltip(self):
        # Build a clean 525x1056 storage ROI with 4 rows separated every 127px.
        frame = numpy.full((525, 1056, 3), 245, dtype=numpy.uint8)
        for y in (0, 127, 254, 381):
            frame[y:y + 3, :] = (210, 225, 230)
        # Put a compressed teal tooltip pill (BGR 159, 176, 61) in the gap below row 1 (y=127..254 -> gap 243..254).
        frame[243:254, 200:460] = (159, 176, 61)
        rows = list(storage._parse_frame(frame))
        # Row 1 (y=127) must be skipped; the other 3 rows remain.
        self.assertEqual(len(rows), 3)


class ScannerExtraTest(parameterized.TestCase):

    @parameterized.named_parameters([
        (k.replace('.', '_'), k, v) for k, v in EXTRA_GROUND_TRUTH.items()
    ])
    def test_extra(self, filename, expected):
        filepath = os.path.join('examples/extra', filename)
        try:
            results = scanner.scan_media(filepath)
            actual = results.items
            if isinstance(expected, list):
                self.assertEqual(results.unmatched, [])
        except AssertionError as e:
            actual = str(e)
        self.assertEqual(expected, actual)


class ScannerLocaleTest(parameterized.TestCase):

    @parameterized.named_parameters([
        (k.replace('.', '_'), k, v) for k, v in LOCALE_CASES.items()
    ])
    def test_locale_catalog(self, filename, case):
        filepath = os.path.join('examples/extra', filename)
        results = scanner.scan_media(filepath, locale='auto')
        self.assertEqual(results.mode, ScanMode.CATALOG)
        self.assertEqual(results.locale, case['expected_locale'])
        self.assertGreaterEqual(len(results.items), case['min_items'])
        self.assertLessEqual(len(results.unmatched), case['max_unmatched'])
        for required_item in case['required_items']:
            self.assertIn(required_item, results.items)


if __name__ == '__main__':
    absltest.main()
