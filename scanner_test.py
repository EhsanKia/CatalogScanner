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
                '불', '분추', '이스터 풍선 6',
            },
            locale='ko-kr',
        )
        for expected in ('TOY 오디오', 'TOY 피규어', '닻 장식', '빨랫대야', '짐 볼'):
            self.assertIn(expected, ko_matched)
        self.assertEqual(ko_unmatched, ['분추', '불', '이스터 풍선 6'])

        tw_matched, tw_unmatched = catalog.match_items(
            {
                '冰桌', '冰桶', '冰箱', '和服架', '飛鏢靶',
                '樸春DIY作業台', '穴烤爐', '波斯菊花決',
                '復活節廟子', '火火',
            },
            locale='zh-tw',
        )
        for expected in ('樸素DIY作業台', '波斯菊花灑', '窯烤爐'):
            self.assertIn(expected, tw_matched)
        self.assertEqual(tw_unmatched, ['復活節廟子', '火火'])

        cn_matched, cn_unmatched = catalog.match_items(
            {
                '书堆', '冰桶', '户外长凳', '电子闹钟', '黄金柏',
                '青蔡庭院五', '水果蓝', '公园攀仆加',
                '复活节受子',
            },
            locale='zh-cn',
        )
        for expected in ('公园攀爬架', '水果篮', '青苔庭院石'):
            self.assertIn(expected, cn_matched)
        self.assertEqual(cn_unmatched, ['复活节受子'])

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
            for width, height in ((854, 480), (640, 360)):
                path = os.path.join(tmpdir, f'low_{width}x{height}.png')
                cv2.imwrite(path, numpy.zeros((height, width, 3), dtype=numpy.uint8))
                with self.assertRaisesRegex(AssertionError, f'Invalid resolution: {width}x{height}'):
                    scanner.scan_media(path)

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
