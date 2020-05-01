# CatalogScanner
Scans Animal Crossing: New Horizon catalog from video of user scrolling through.

This requires Python 3 and [Tesseract-OCR](https://tesseract-ocr.github.io/tessdoc/Home.html) to work.
The Python library requirements can be installed through the requirements.txt

It can be used by page scrolling through the item catalog in Nook Shopping (by holding down right analog stick),
and holding the Capture button at the end to record the previous 30s. You can trim the video with the Switch editor
to only get the part where you are scrolling to speed up processing time. An example video is provided (catalog.mp4).

You can then extract the video with an SD card or through social media. You call this script with the video file name
and it will output the name of all the items to the console.

The item name data comes from https://github.com/imthe666st/ACNH
