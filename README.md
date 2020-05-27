# CatalogScanner
Script to scan video of user scrolling through their AC:NH Nook Shop catalog.
This is the source code for the Twitter bot [@CatalogScanner](https://twitter.com/CatalogScanner), which is an automated version of this.

This requires Python 3 and [Tesseract-OCR](https://tesseract-ocr.github.io/tessdoc/Home.html) to work.
On the Python you can install, you can install the required libraries using pip and requirements.txt:

```shell
cd CatalogScanner
python3 -m venv .env
.env/Scripts/activate
pip install -r requirements.txt
python scanner.py catalog.mp4
```

You can use `--for_sale` to filter out items that are not purchasable.
For non-English catalogs, you can use `--lang` to adjust the parsed language.

It can be used by page scrolling through the item catalog in Nook Shopping (by holding down right analog stick),
and holding the Capture button at the end to record the previous 30s. You can trim the video with the Switch editor
to only get the part where you are scrolling to speed up processing time. An example video is provided (catalog.mp4).

You can then extract the video with an SD card or through social media. You call this script with the video file name
and it will output the name of all the items to the console.

The item name data comes from https://github.com/imthe666st/ACNH
