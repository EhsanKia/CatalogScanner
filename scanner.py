from common import ScanResult
import catalog
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
    else:
        return recipes.scan_recipes(file_name, locale=locale)


def _detect_video_type(file_name: str) -> str:
    video_capture = cv2.VideoCapture(file_name)

    # Check the first ~3s of the video.
    for _ in range(100):
        success, frame = video_capture.read()
        if not success or frame is None:
            raise AssertionError('Unable to parse video.')

        assert frame.shape[:2] == (720, 1280), \
            'Invalid resolution: {1}x{0}'.format(*frame.shape)

        # Get the average color of the background.
        color = frame[300:400, :10].mean(axis=(0, 1))
        if numpy.linalg.norm(color - catalog.BG_COLOR) < 10:
            return 'catalog'
        elif numpy.linalg.norm(color - recipes.BG_COLOR) < 10:
            return 'recipes'

    raise AssertionError('Video is not showing catalog or recipes.')


def main(argv):
    if len(argv) > 1:
        video_file = argv[1]
    elif FLAGS.mode == 'recipes':
        video_file = 'videos/diy.mp4'
    else:
        video_file = 'videos/catalog.mp4'

    result = scan_video(
        video_file,
        mode=FLAGS.mode,
        locale=FLAGS.locale,
        for_sale=FLAGS.for_sale,
    )

    result_count, result_mode = len(result.items), result.mode.name.lower()
    print(f'Found {result_count} items in {result_mode} [{result.locale}]')
    print('\n'.join(result.items))


if __name__ == "__main__":
    app.run(main)
