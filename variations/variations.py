import datetime
import difflib
import json
import os
import sys

from dataclasses import dataclass
from typing import Optional, Sequence, Set

import cv2
import numpy
from absl import app, flags
from PIL import Image
from tesserocr import PSM, PyTessBaseAPI

flags.DEFINE_integer('device_id', None, 'ID of the capture card device.')
flags.DEFINE_boolean('for_sale', False, 'Whether to only process items that are for sale.')
flags.DEFINE_string('video_path', None, 'Path to video file to use instead of capture device.')

FLAGS = flags.FLAGS
TUTORIAL_LINES = [
    'Q - quit scanner',
    'S - save items',
    'R - reset items',
    'F - for_sale filter',
]


@dataclass
class Rectangle:
    x1: int = 0
    x2: int = 0
    y1: int = 0
    y2: int = 0

    @property
    def p1(self):
        return self.x1, self.y1

    @property
    def p2(self):
        return self.x2, self.y2

    @property
    def slice(self):
        return slice(self.y1, self.y2), slice(self.x1, self.x2)


class VariationParser:

    def __init__(self):
        self.tesseract = PyTessBaseAPI(path='./', psm=PSM.SINGLE_LINE)
        with open('en-us-var.json', encoding='utf-8') as fp:
            self.item_db = json.load(fp)

        self.items: Set[str] = set()
        self.active_section = 0
        self.section_name = None
        self.for_sale = False

        self._tesseract_cache = {}
        self._item_cache = {}

    def annotate_frame(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Parses various parts of a frame for catalog items and annotates it."""

        # Detect whether we are in in Nook Shopping catalog.
        if numpy.linalg.norm(frame[500, 20] - (182, 249, 255)) > 5:
            text = 'Navigate to Nook catalog to start!'
            opts = {'org': (200, 70), 'fontFace': cv2.FONT_HERSHEY_PLAIN,
                    'lineType': cv2.LINE_AA, 'fontScale': 3}
            frame = cv2.putText(frame, text, color=(0, 0, 0), thickness=7, **opts)
            return cv2.putText(frame, text, color=(100, 100, 255), thickness=3, **opts)

        # Show controls on screen.
        cv2.rectangle(frame, (0, 0), (300, 130), (106, 226, 240), -1)
        for i, line in enumerate(TUTORIAL_LINES):
            if line.startswith('F'):
                line += ' (%s)' % ('ON' if self.for_sale else 'OFF')
            frame = cv2.putText(frame, line, (30, 25 + i * 30), 0, 0.8, 0, 2)

        # Show user the item count at the bottom.
        count_text = 'Item count: %d' % len(self.items)
        if not self.section_name:
            count_text = 'Items saved to disk'
        frame = cv2.putText(frame, count_text, (500, 700), 0, 1, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        section = numpy.where((154 < gray[20, 250:]) & (gray[20, 250:] < 162))[0]
        if section.any() and abs(self.active_section - section[0]) > 10:
            # Grab the new section name
            x1, *_, x2 = 250 + section
            section_region = 255 - gray[8:32, x1+5:x2-5]
            self.section_name = self.image_to_text(section_region)

            # Reset item selection on section change
            self.active_section = section[0]
            self.items = set()
        elif not self.active_section:
            return frame  # Return early if not section is found.

        item_name = None
        variation_name = None

        selected = self.get_selected_item(frame)
        if not selected:  # Quit early if not item is selected
            return frame

        price_region = gray[selected.y1:selected.y2, 1070:1220]
        if self.for_sale and price_region.min() > 100:
            # Skip items not for sale
            p1, p2 = (selected.x1, selected.y1 + 20), (selected.x2, selected.y1 + 20)
            return cv2.line(frame, p1, p2, color=(0, 0, 255), thickness=2)

        # Parse item name and display a rectangle around it.
        item_name = self.image_to_text(gray[selected.slice])
        frame = cv2.putText(frame, item_name, selected.p1, 0, 1, 0)

        # Parse variation and draw rectangle around it if there is one.
        variation = self.get_variation(gray)
        if variation:
            frame = cv2.rectangle(frame, variation.p1, variation.p2, 0)
            variation_name = self.image_to_text(gray[variation.slice])
            frame = cv2.putText(frame, variation_name, variation.p1, 1, 2, 0)

        # Match the name and optional variation against database and register it.
        full_name = self.resolve_name(item_name, variation_name)
        if full_name:
            self.items.add(full_name)

        return frame

    def get_selected_item(self, frame: numpy.ndarray) -> Optional[Rectangle]:
        """Returns the rectangle around the selected item name if there is one."""
        # Search for the yellow selected region along the item list area.
        select_region = numpy.where(frame[140:640, 1052, 0] < 100)[0]
        if not select_region.any():
            return None

        rect = Rectangle(x1=635, x2=1050)

        # Find the top/bottom boundaries of the selected area
        rect.y1 = 140 + select_region[0] + 8
        rect.y2 = 140 + select_region[-1] - 4

        if rect.y2 - rect.y1 < 35:
            return None

        # Detect to width of the name by collapsing along the x axis
        # and finding the right-most dark pixel (text).
        item_region = frame[rect.y1:rect.y2, rect.x1:rect.x2, 1]
        detected_text = numpy.where(item_region.min(axis=0) < 55)[0]
        if not detected_text.any():
            return None

        rect.x2 = 635 + detected_text[-1] + 10
        return rect

    def get_variation(self, gray: numpy.ndarray) -> Optional[Rectangle]:
        """Returns the rectangle around the variation text if there is one."""
        # There's a white box if the item has a variation.
        if gray[650, 25] < 250:
            return None

        variation = Rectangle(x1=30, y1=628, y2=665)
        # Find the width of tqhe variation box by horizontal floodfill.
        variation.x2 = numpy.argmax(gray[variation.y1, :] < 250) - 15
        return variation

    def resolve_name(self, item: Optional[str], variation: Optional[str]) -> Optional[str]:
        """Resolves an item and optional variation name against the item database."""
        key = (item, variation)
        if key not in self._item_cache:
            item = best_match(item, self.item_db)
            variation = best_match(variation, self.item_db.get(item))
            if variation:
                self._item_cache[key] = f'{item} [{variation}]'
            elif item and not self.item_db[item]:
                self._item_cache[key] = item
            else:
                self._item_cache[key] = None
        return self._item_cache[key]

    def image_to_text(self, text_area: numpy.ndarray) -> str:
        """Runs OCR over a given image and returns the parsed text."""
        img_hash = str(cv2.img_hash.averageHash(text_area)[0])
        if img_hash not in self._tesseract_cache:
            image = Image.fromarray(text_area)
            self.tesseract.SetImage(image)
            text = self.tesseract.GetUTF8Text().strip()
            self._tesseract_cache[img_hash] = text
        return self._tesseract_cache[img_hash]

    def save_items(self) -> None:
        """"Saves the collected items to a text file on disk and clears list."""
        if not self.items or not self.section_name:
            return

        date = datetime.datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        with open(f'{self.section_name} ({date}).txt', 'w') as fp:
            print(f'Saving {len(self.items)} items to {fp.name}')
            fp.write('\n'.join(sorted(self.items)))

        self.section_name = None
        self.items = set()


def best_match(needle: Optional[str], haystack: Sequence[str]) -> Optional[str]:
    """Finds the closest match of a given string in a list of potential strings."""
    if not needle or not haystack:
        return None

    matches = difflib.get_close_matches(needle, haystack, n=1)
    return matches[0] if matches else None  # type: ignore


def pick_device_id() -> int:
    """Tries to use a library to list video devices"""
    try:
        import pymf  # noqa
    except ImportError:
        return 0

    device_list = pymf.get_MF_devices()
    if not device_list:
        raise RuntimeError('No video devices found...')
    elif len(device_list) == 1:
        return 0  # Only one choice...

    print('Detected devices:')
    for device_id, device_name in enumerate(device_list):
        print(f'  {device_id}) {device_name}')

    while True:
        pick = input('Pick a device number: ')
        if pick.isdigit() and 0 <= int(pick) < len(device_list):
            break
        print('Invalid id, pick from the list above')
    return int(pick)


def main(argv):
    # Handle working inside PyInstaller
    script_dir = os.path.dirname(argv[0])
    data_dir = getattr(sys, '_MEIPASS', script_dir)
    os.chdir(data_dir or '.')

    # Find the right device ID.
    if FLAGS.video_path:
        video_capture = FLAGS.video_path
        print(f'Processing video file {video_capture}')
    elif FLAGS.device_id is not None:
        video_capture = FLAGS.device_id
    else:
        video_capture = pick_device_id()

    # Connect to input feed and adjust video dimensions.
    cap = cv2.VideoCapture(video_capture)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    parser = VariationParser()
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = parser.annotate_frame(frame)
        if FLAGS.video_path:
            continue  # Skip showing frame if parsing video.

        cv2.imshow('frame', frame)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q'):
            break
        if keypress == ord('s'):
            parser.save_items()
        if keypress == ord('r'):
            parser.items = set()
        if keypress == ord('f'):
            parser.for_sale = not parser.for_sale

    # Save any remaining unsaved items.
    parser.save_items()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
