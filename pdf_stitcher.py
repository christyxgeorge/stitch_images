"""
Stitch 2 pages from PDF files or 2 images (jpg/png).
For vertical stitching, rotate the images/pages appropriately before stitching.

Other tools tried:
OpenCV Stitcher does not work. Neither does the 'stitching' pip package

Note:
1. Used # type: ignore for ignoring Pyright errors
2. Converted the code to work with opencv-python==4.8

References:
1. https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
2. https://stackoverflow.com/questions/46176380/key-point-detection-and-image-stitching
"""
from pdf2image import convert_from_path
import cv2
import numpy as np
import argparse
import os
from typing import Any

class Stitcher:
    def __init__(self, file1, file2=None, page1=0, page2=0, concatenate=False, vertical=False):
        self.root_dir = os.path.dirname(os.path.abspath(file1))
        self.mode = 'concatenate' if concatenate else 'stitch'
        self.horizontal = False if vertical else True
        print(f"Mode: {self.mode}, Root Dir = {self.root_dir}, File = {file1}")
        self.img_0, self.img_1 = self.setup_images(file1, file2, page1, page2)
        self.output_file = self.get_output_filename(file1, file2, page1, page2)
        self.show_keypoint_matches = False

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
                            prog='pdf_stitcher',
                            description='Stitches 2 pages of a PDF file  (or 2 image files)')
        parser.add_argument('-c', '--concatenate', action='store_true', help='Concatenate images - No stitching')
        parser.add_argument('-f', '--file', dest='file1', required=True, help='File name (PDF or an  image)')
        parser.add_argument('--file2', help='Second Image if first file is an image')
        parser.add_argument('--page1', type=int, default=0, help='First page of the PDF file')
        parser.add_argument('--page2', type=int, default=0, help='Second page of the PDF file')
        parser.add_argument('-v', '--vertical', action='store_true', help='Supported only for concatenate')
        args = parser.parse_args()
        print(f"Arguments = {args}")
        if args.file1.endswith('.pdf'): # PDF File
            if not args.page1 and not args.page2:
                print("Need to specify page1 and page2 for PDF stitching")
                exit(-1)
        elif not args.file2:
                print("Need to specify second image file")
                exit(-2)
        return args

    def get_output_filename(self, file1, file2, page1, page2):
        _base_dir = os.path.dirname(file1)
        _base_name = os.path.basename(file1)
        _base_name = os.path.splitext(_base_name)[0]
        return os.path.join(_base_dir, f"{_base_name}_p{page1}{page2}.png")

    def setup_images(self, file1, file2, page1, page2):
        if file1.endswith('.pdf'): # PDF File
            pages = convert_from_path(file1)
            print(f"Number of pages in the PDF file = {len(pages)}")
            if page1 >= len(pages) or page2 >= len(pages):
                print(f"Page {page1} or Page {page2} is more than the number of pages {len(pages)} ")
                exit(-2)
            img_0 = np.array(pages[page1])
            img_1 = np.array(pages[page2])
        else: # Image files
            img_0 = cv2.imread(file1)
            img_1 = cv2.imread(file2)
        if not self.horizontal and self.mode == 'stitch':
            img_0 = cv2.rotate(img_0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_1 = cv2.rotate(img_1, cv2.ROTATE_90_CLOCKWISE)
        print("-- Images to be Processed --")
        print(f"Sizes: {img_0.size} // {img_1.size}")
        print(f"Shapes: {img_0.shape} // {img_1.shape}")
        return img_0, img_1

    def process(self):
        if self.mode == 'stitch' and self.show_keypoint_matches:
            (result, vis) = self.stitch(showMatches=True)
            cv2.imshow("Keypoint Matches", vis)
        elif self.mode == 'stitch':
            result = self.stitch()
        else:
            result = self.concatenate()

        if result.any():
            print()
            if self.mode == 'stitch':
                print(f"Writing to file: {self.output_file}")
                cv2.imwrite(self.output_file, result)
            # opencv code to view image
            img = cv2.resize(result, None, fx=0.5, fy=0.5)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def stitch(self, ratio=0.75, reprojThresh=4.0, showMatches=False) -> Any:
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = (self.img_0, self.img_1)

        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
            featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None
           # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        # print(f"Sizes: {imageA.size[1]} / {imageB.size[1]} / {imageA.size[0]}")
        result = cv2.warpPerspective(imageA, H,
            (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # if vertical, we turn it clockwise to make it vertical!
        if not self.horizontal:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)

        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
                status)

            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        descriptor = cv2.SIFT_create() # type: ignore
        (kps, features) = descriptor.detectAndCompute(gray, None)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps]) # type: ignore

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
        ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce") # type: ignore
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches]) # type: ignore
            ptsB = np.float32([kpsB[i] for (i, _) in matches]) # type: ignore

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis

    def concatenate(self):
        if self.horizontal: # concate horizontally
            img = np.concatenate((self.img_0, self.img_1), axis=1)
        else:
            img = np.concatenate((self.img_0, self.img_1), axis=0)
        return img

# Main Logic: Execute when the module is not initialized from an import statement.
if __name__ == '__main__':
    args = Stitcher.parse_args()
    stitcher = Stitcher(
        args.file1, args.file2, args.page1, args.page2, concatenate=args.concatenate, vertical=args.vertical
    )
    stitcher.process()
