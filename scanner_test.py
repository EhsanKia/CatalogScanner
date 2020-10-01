import json
import os

from absl import logging
from absl.testing import absltest

import scanner
from common import ScanMode


class ScannerTest(absltest.TestCase):

    def setUp(self):
        with open('testdata/examples.json', encoding='utf-8') as fp:
            self.ground_truth = json.load(fp)

    @property
    def name(self):
        return self.id().split('.')[-1]

    def test_catalog(self):
        results = scanner.scan_media('examples/catalog.mp4')
        self.assertEqual(results.mode, ScanMode.CATALOG)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_catalog_forsale(self):
        results = scanner.scan_media('examples/catalog.mp4', for_sale=True)
        self.assertEqual(results.mode, ScanMode.CATALOG)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_recipes(self):
        results = scanner.scan_media('examples/recipes.mp4')
        self.assertEqual(results.mode, ScanMode.RECIPES)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_recipes_translate(self):
        results = scanner.scan_media('examples/recipes.mp4', locale='fr-eu')
        self.assertEqual(results.mode, ScanMode.RECIPES)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'fr-eu')

    def test_critters(self):
        results = scanner.scan_media('examples/critters.mp4')
        self.assertEqual(results.mode, ScanMode.CRITTERS)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_critters_translate(self):
        results = scanner.scan_media('examples/critters.mp4', locale='ko-kr')
        self.assertEqual(results.mode, ScanMode.CRITTERS)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'ko-kr')

    def test_reactions(self):
        results = scanner.scan_media('examples/reactions.jpg')
        self.assertEqual(results.mode, ScanMode.REACTIONS)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_reactions_translate(self):
        results = scanner.scan_media('examples/reactions.jpg', locale='de-eu')
        self.assertEqual(results.mode, ScanMode.REACTIONS)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'de-eu')

    def test_music(self):
        results = scanner.scan_media('examples/music.mp4')
        self.assertEqual(results.mode, ScanMode.MUSIC)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'en-us')

    def test_music_translate(self):
        results = scanner.scan_media('examples/music.mp4', locale='ja-jp')
        self.assertEqual(results.mode, ScanMode.MUSIC)
        self.assertEqual(results.items, self.ground_truth[self.name])
        self.assertEqual(results.locale, 'ja-jp')

    def test_extra(self):
        with open('testdata/extra.json', encoding='utf-8') as fp:
            ground_truth = json.load(fp)

        for filename in ground_truth:
            logging.info('Testing %r', filename)
            filepath = os.path.join('examples/extra', filename)
            try:
                results = scanner.scan_media(filepath)
                actual = results.items
            except AssertionError as e:
                actual = str(e)

            self.assertEqual(actual, ground_truth[filename])


if __name__ == "__main__":
    absltest.main()
