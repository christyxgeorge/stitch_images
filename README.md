
# Stitch 2 pages from PDF files or 2 images (jpg/png).

Useful for stitching pages that are larger than the scanner can handle. Scan into multiple pages and use this to stitch them together. In some cases, we may need to stitch multiple pages into separate image files and then stitch those images to create the final image. Have tried with 6 pages (2 x 3) scanned image and worked flawlessly!

For vertical stitching, rotate the images/pages appropriately before stitching.


# Dependencies
install poppler-cpp (using macports or brew) - Needed by python-poppler
install requirements from the requirements.cpp file

# Usage
usage: pdf_stitcher [-h] [-c] -f FILE [--file2 FILE2] [--page1 PAGE1] [--page2 PAGE2] [-v]

Stitches 2 pages of a PDF file (or 2 image files)

options:
  -h, --help            show this help message and exit
  -c, --concatenate     Concatenate images - No stitching
  -f FILE, --file FILE  File name (PDF or an image)
  --file2 FILE2         Second Image if first file is an image
  --page1 PAGE1         First page of the PDF file
  --page2 PAGE2         Second page of the PDF file
  -v, --vertical        Supported only for concatenate

# References:
1. https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
2. https://stackoverflow.com/questions/46176380/key-point-detection-and-image-stitching

# Notes
1. Used # type: ignore for ignoring Pyright errors
2. Converted the code to work with opencv-python==4.8
3. OpenCV Stitcher does not work. Neither does the 'stitching' pip package
