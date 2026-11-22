import json
import os
import shutil
import tempfile

import numpy
from absl.testing import absltest, parameterized

import catalog
import music
import scanner
from common import ScanMode, is_image_file


def _fetch_ground_truth():
    with open('testdata/examples.json', encoding='utf-8') as fp:
        yield json.load(fp)
    with open('testdata/extra.json', encoding='utf-8') as fp:
        yield json.load(fp)


GROUND_TRUTH, EXTRA_GROUND_TRUTH = _fetch_ground_truth()


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
            {'Writing desk', 'Teacup ride', 'Fish and chips', 'Tee with silicon bib'},
            locale='en-us',
        )
        self.assertEqual(
            matched,
            ['Fish-and-chips', 'Plaza teacup ride', 'Study desk', 'Tee with silicone bib'],
        )
        self.assertEqual(unmatched, [])

        # Short noisy CJK strings must not false-match unrelated items.
        ja_matched, ja_unmatched = catalog.match_items(
            {'アーガイルベスト', 'あきいろチェックワンピース', 'あさがおのゆかた', 'アノラック', 'ごの'},
            locale='ja-jp',
        )
        self.assertIn('ごの', ja_unmatched)
        self.assertNotIn('みの', ja_matched)

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

    def test_music_phash_rejects_noise_cover(self):
        rng = numpy.random.default_rng(42)
        noise_cover = rng.integers(0, 256, size=(260, 260, 3), dtype=numpy.uint8)
        matched, unmatched = music.match_songs([noise_cover])
        self.assertEqual(matched, [])
        self.assertEqual(len(unmatched), 1)


class ScannerExtraTest(parameterized.TestCase):

    @parameterized.named_parameters([
        (k.replace('.', '_'), k, v) for k, v in EXTRA_GROUND_TRUTH.items()
    ])
    def test_extra(self, filename, expected):
        filepath = os.path.join('examples/extra', filename)
        try:
            results = scanner.scan_media(filepath)
            actual = results.items
        except AssertionError as e:
            actual = str(e)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    absltest.main()
