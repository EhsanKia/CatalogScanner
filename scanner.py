import cv2
import numpy

import catalog
import recipes

from absl import app
from absl import flags

from typing import List

CATALOG_COLOR = numpy.array([181, 253, 253])
RECIPES_COLOR = numpy.array([195, 223, 228])

FLAGS = flags.FLAGS
flags.DEFINE_enum('locale', 'auto', catalog.LOCALE_MAP.keys(),
                  'The locale to use for parsing item names.')
flags.DEFINE_bool('for_sale', None,
                  'If true, the scanner will skip items that are not for sale.')
flags.DEFINE_enum('mode', 'auto', ['auto', 'catalog', 'recipes'],
                  'The type of video to scan. Catalog refers to Nook shopping catalog '
                  'and recipes refers to DIY list. Auto tries to detect from the video frames.')


def scan_video(file_name: str, mode: str = 'auto', locale: str = 'auto',
               for_sale: bool = False) -> List[str]:
    if mode == 'auto':
        mode = _detect_video_type(file_name)

    if mode == 'catalog':
        return catalog.scan_catalog(file_name, locale=locale, for_sale=for_sale)
    else:
        return recipes.scan_recipes(file_name, locale=locale)


def _detect_video_type(file_name: str) -> str:
    video_capture = cv2.VideoCapture(file_name)
    success, frame = video_capture.read()
    if not success or frame is None:
        raise AssertionError('Unable to parse video.')

    assert frame.shape[:2] == (720, 1280), \
        'Invalid resolution: {1}x{0}'.format(*frame.shape)

    # Get the average color of the background.
    color = frame[300:400, :10].mean(axis=(0, 1))
    if numpy.linalg.norm(color - CATALOG_COLOR) < 10:
        return 'catalog'
    elif numpy.linalg.norm(color - RECIPES_COLOR) < 10:
        return 'recipes'

    raise AssertionError('Video is not showing catalog or recipes.')


def main(argv):
    if len(argv) > 1:
        video_file = argv[1]
    elif FLAGS.mode == 'recipes':
        video_file = 'diy.mp4'
    else:
        video_file = 'catalog.mp4'

    all_items = scan_video(
        video_file,
        mode=FLAGS.mode,
        locale=FLAGS.locale,
        for_sale=FLAGS.for_sale,
    )
    print('\n'.join(all_items))


if __name__ == "__main__":
    app.run(main)
