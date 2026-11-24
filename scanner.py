from common import ScanResult, is_image_file, to_720p
from typing import Any, Dict, Optional

import catalog
import critters
import music
import reactions
import recipes
import storage

import cv2

from absl import app
from absl import flags
from absl import logging

MAX_VIDEO_DURATION_SECONDS = 180
MIN_FRAME_HEIGHT = 472

SCANNERS: Dict[str, Any] = {
    'catalog': catalog,
    'recipes': recipes,
    'critters': critters,
    'reactions': reactions,
    'music': music,
    'storage': storage,
}

FLAGS = flags.FLAGS
flags.DEFINE_enum('locale', 'auto', list(catalog.LOCALE_MAP),
                  'The locale to use for parsing item names.')
flags.DEFINE_bool('for_sale', False,
                  'If true, the scanner will skip items that are not for sale.')
flags.DEFINE_enum('mode', 'auto', ['auto'] + list(SCANNERS),
                  'The type of catalog to scan. Auto tries to detect from the media frames.')


def _check_video_duration(filename: str) -> None:
    """Ensures video attachments do not exceed the maximum supported duration."""
    if is_image_file(filename):
        return
    video_capture = cv2.VideoCapture(filename)
    try:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 1 and frame_count > 0 and (frame_count / fps) > MAX_VIDEO_DURATION_SECONDS:
            raise AssertionError('Your video is too long.')
    finally:
        video_capture.release()


def scan_media(filename: str, mode: str = 'auto', locale: str = 'auto', for_sale: bool = False) -> ScanResult:
    _check_video_duration(filename)

    if mode == 'auto':
        mode = _detect_media_type(filename)
        logging.info('Detected scan mode: %s', mode)

    if mode not in SCANNERS:
        raise RuntimeError('Invalid mode: %r' % mode)

    assert mode != 'storage', 'Storage scanning is not supported.'

    kwargs = {}
    if mode == 'catalog':
        kwargs['for_sale'] = for_sale

    return SCANNERS[mode].scan(filename, locale=locale, **kwargs)


def _detect_media_type(filename: str) -> str:
    video_capture = cv2.VideoCapture(filename)
    try:
        unsupported_error: Optional[AssertionError] = None

        # Check the first 100 frames for a match.
        for _ in range(100):
            success, frame = video_capture.read()
            if not success or frame is None:
                break

            # Resolution must be at least 480p (allowing 478p mobile/WhatsApp aspect rounding).
            assert frame.shape[0] >= MIN_FRAME_HEIGHT, \
                'Invalid resolution: {1}x{0}'.format(*frame.shape)

            # Resize frames to 720p since matching is built for 720p.
            # This is to handle 1080p videos from the Switch 2.
            frame = to_720p(frame)

            for mode, scanner in SCANNERS.items():
                try:
                    if scanner.detect(frame):
                        return mode
                except AssertionError as e:
                    if unsupported_error is None:
                        unsupported_error = e

        if unsupported_error is not None:
            raise unsupported_error
        raise AssertionError('Media is not showing a known scan type.')
    finally:
        video_capture.release()


def main(argv):
    if len(argv) > 1:
        media_file = argv[1]
    elif FLAGS.mode == 'recipes':
        media_file = 'examples/recipes.mp4'
    elif FLAGS.mode == 'critters':
        media_file = 'examples/critters.mp4'
    elif FLAGS.mode == 'reactions':
        media_file = 'examples/reactions.jpg'
    elif FLAGS.mode == 'music':
        media_file = 'examples/music.mp4'
    else:
        media_file = 'examples/catalog.mp4'

    result = scan_media(
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
