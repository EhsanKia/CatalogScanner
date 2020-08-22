from common import ScanResult
import catalog
import critters
import recipes

import cv2
import numpy

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_enum('locale', 'auto', catalog.LOCALE_MAP.keys(),
                  'The locale to use for parsing item names.')
flags.DEFINE_bool('for_sale', None,
                  'If true, the scanner will skip items that are not for sale.')
flags.DEFINE_enum('mode', 'auto', ['auto', 'catalog', 'recipes'],
                  'The type of video to scan. Catalog refers to Nook shopping catalog '
                  'and recipes refers to DIY list. Auto tries to detect from the video frames.')


def scan_video(file_name: str, mode: str = 'auto', locale: str = 'auto',
               for_sale: bool = False) -> ScanResult:
    if mode == 'auto':
        mode = _detect_video_type(file_name)
        logging.info('Detected video mode: %s', mode)

    if mode == 'catalog':
        return catalog.scan_catalog(file_name, locale=locale, for_sale=for_sale)
    elif mode == 'recipes':
        return recipes.scan_recipes(file_name, locale=locale)
    elif mode == 'critters':
        return critters.scan_critters(file_name, locale=locale)
    else:
        raise RuntimeError('Invalid mode: %r' % mode)


def _detect_video_type(file_name: str) -> str:
    video_capture = cv2.VideoCapture(file_name)

    # Check the first ~3s of the video.
    for _ in range(100):
        success, frame = video_capture.read()
        if not success or frame is None:
            break

        assert frame.shape[:2] == (720, 1280), \
            'Invalid resolution: {1}x{0}'.format(*frame.shape)

        # Get the average color of the background.
        color = frame[:20, 1100:1150].mean(axis=(0, 1))
        if numpy.linalg.norm(color - catalog.BG_COLOR) < 5:
            return 'catalog'
        elif numpy.linalg.norm(color - recipes.BG_COLOR) < 5:
            return 'recipes'
        elif numpy.linalg.norm(color - critters.BG_COLOR) < 5:
            return 'critters'

    raise AssertionError('Video is not showing a known scan type.')


def main(argv):
    if len(argv) > 1:
        media_file = argv[1]
    elif FLAGS.mode == 'recipes':
        media_file = 'examples/recipes.mp4'
    elif FLAGS.mode == 'critters':
        media_file = 'examples/critters.mp4'
    elif FLAGS.mode == 'reactions':
        media_file = 'examples/reactions.jpg'
    else:
        media_file = 'examples/catalog.mp4'

    result = scan_video(
        media_file,
        mode=FLAGS.mode,
        locale=FLAGS.locale,
        for_sale=FLAGS.for_sale,
    )

    result_count, result_mode = len(result.items), result.mode.name.lower()
    print(f'Found {result_count} items in {result_mode} [{result.locale}]')
    print('\n'.join(result.items))


if __name__ == "__main__":
    app.run(main)
