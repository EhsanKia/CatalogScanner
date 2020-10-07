# CatalogScanner [![Python version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Script to scan video of user scrolling through their AC:NH Nook Shop catalog or DIY recipes list.

This is the source code for the Twitter bot [@CatalogScanner](https://twitter.com/CatalogScanner), which is an automated version of this.

## Installation

This script requires [Python 3](https://www.python.org/downloads/release/python-377/) and [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract/wiki) to work.

You can then install the required libraries using pip and requirements.txt:

```shell
cd CatalogScanner
python3 -m venv .env
.env/Scripts/activate
pip install -r requirements.txt
```

## Usage

Before using this script, you need to create a valid video file. There are many types of supported media.
For the full list alongside instructions for each, check out <https://nook.lol>.

### Scanning the media

Once you have your video file, you can pass it as the first argument to the script:

```sh
python scanner.py catalog.mp4
```

If you have screenshots, you can pass it as is if there is a single one, otherwise you need
to number them starting with 0 and pass the filename with `%d` instead of the numbers.

```sh
python scanner.py catalog_%d.png
```

By default, it will detect the media type (catalog, recipes, etc), but you can force on with `--mode`.

You can use `--for_sale` to filter out items that are not purchasable,
and you can use `--locale` to adjust the parsed language.

By default, the script prints out the name of all the items found in your catalog video.

## Credits

The item name data comes from:

- <https://tinyurl.com/acnh-sheet>
- <https://github.com/imthe666st/ACNH>
- <https://github.com/alexislours/translation-sheet-data>
