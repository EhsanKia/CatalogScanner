# CatalogScanner
Script to scan video of user scrolling through their AC:NH Nook Shop catalog.

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

Before using this script, you need to create a valid video file. The instructions are:
1. Open Nook Shopping (ATM or phone app)
2. Select desired catalog and section
3. Scroll to the bottom using **right analog stick** (page scrolling)
4. Hold down "Capture" button on left joycon to record the last 30s
5. Go to Switch's Album gallery and select your video
6. Press A and select "Trim"
7. Cut the video to the start/end of the scrolling and save
8. Export the video either through Twitter or with an SDcard

You can then pass the video filename as the first argument to the script:

```
python scanner.py catalog.mp4
```

You can use `--for_sale` to filter out items that are not purchasable,
and you can use `--lang` to adjust the parsed language.

By default, the script prints out the name of all the items found in your catalog video.


## Credits

The item name data comes from https://github.com/imthe666st/ACNH
