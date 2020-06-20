# Variations Scanner

This is a manual version of the Catalog Scanner which does support variations.

**It requires a capture card and currently only works with enUS.** It has also only been tested with my capture card (Elgato HD60 S) on Windows 10.

## Pre-built binary

For windows, there's a binary version under [Releases](https://github.com/EhsanKia/CatalogScanner/releases). It's a single binary with everything included. This has only been tested on my device, run at your own discretion.

## Manual installation

Simply grab the folder, install the dependencies and run `variations.py`. Unlike `scanner.py`, this uses [tesserocr](https://github.com/sirfz/tesserocr) (not pytesseract). That technically means that you don't need to have Tesseract-OCR installed, the library comes with all the required modules, and `eng.traineddata` is included in the directory for simplicity. On Windows, you can grab the [pre-built version of tesserocr](https://github.com/simonflueckiger/tesserocr-windows_build/releases) and
install it using `pip install <filename>.whl`. The rest of the libraries can be installed using `pip install -r requirements.txt`.

## Usage

Once you launch this program, it will try to connect to your capture card and show you a view of your game.
You then will have to navigate to your Nook Shopping catalog to start the process.
At this point, you have to simply scroll through every item one by one, toggling the variations for each item that has them.
The program will record every item you hover on and show the total count at the bottom of the screen.
If you switch sections, it will reset the count automatically. You can also toggle "For Sale" filter to only add items that are purchasable.
Once you're done, you can save the items which will create text file in the same directory.

The controls are:
 - Q to quit the program
 - R to reset scanned items
 - S to save scanned items
 - F to toggle for_sale filter

By default, the script tries to access your device on port #1. If that does not work or grabs your webcam's feed, you can try different values (0, 1, 2, 3, etc) using the `--device_id` flag.