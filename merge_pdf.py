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
    def __init__(self):
        self.args = self.parse_args()
        self.root_dir = os.path.dirname(os.path.abspath(self.args.file))
        self.mode = 'concatenate' if self.args.concatenate else 'stitch'
        print(f"Mode: {self.mode}, Root Dir = {self.root_dir}, File = {self.args.file}")
        self.img_0, self.img_1 = self.setup_images(self.args)
        self.horizontal = False if self.args.vertical else True

    def setup_images(self, args):
        if args.file.endswith('.pdf'):
            if not args.page1 and not args.page2:
                print("Need to specify page1 and page2")
                exit(-1)
            pages = convert_from_path(args.file)
            print(f"Number of pages in the PDF file = {len(pages)}")
            if args.page1 >= len(pages) or args.page2 >= len(pages):
                print(f"Page {args.page1} or Page {args.page2} is more than the number of pages {len(pages)} ")
                exit(-2)
            img_0 = np.array(pages[args.page1])
            img_1 = np.array(pages[args.page2])
        else:
            if not args.file2:
                print("Need to specify file2")
                exit(-3)
            img_0 = cv2.imread(args.file)
            img_1 = cv2.imread(args.file2)
        print("-- Images --")
        print(f"Sizes: {img_0.size} // {img_1.size}")
        print(f"Shapes: {img_0.shape} // {img_1.shape}")
        return img_0, img_1

    def parse_args(self):
        parser = argparse.ArgumentParser(
                            prog='PDF Stitcher',
                            description='Stitches 2 pages of a PDF file')
        parser.add_argument('-c', '--concatenate', action='store_true')
        parser.add_argument('-f', '--file', required=True)
        parser.add_argument('--file2', help='Second Image if first file is an image')
        parser.add_argument('--page1', type=int, default=0, help='First page of the PDF file')
        parser.add_argument('--page2', type=int, default=0, help='Second page of the PDF file')
        parser.add_argument('-v', '--vertical', action='store_true')
        args = parser.parse_args()
        print(f"Arguments = {args}")
        return args

    def process(self):
        if self.mode == 'stitch':
            (result, vis) = self.stitch(showMatches=True)
            cv2.imshow("Keypoint Matches", vis)
        else:
            result = my_stitcher.concatenate()

        if result.any():
            print()
            if self.mode == 'stitch':
                _base_dir = os.path.dirname(self.args.file)
                _base_name = os.path.basename(self.args.file)
                _file_name = os.path.join(_base_dir, f"{_base_name}_p{self.args.page1}{self.args.page2}.png")
                print(f"Writing to file: {_file_name}")
                cv2.imwrite(_file_name,result)
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
    stitcher = Stitcher()
    stitcher.process()
