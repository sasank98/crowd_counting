import os
import sys
import cv2
import matplotlib
import numpy as np
from ultralytics import YOLO
import time

import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# HEIGHT = 720
# WIDTH = 1280
sift = cv2.SIFT_create()

def compute_matches(kp1, kp2, desc1, desc2, matcher = cv2.BFMatcher(cv2.NORM_L2)):
    """This function computes the matches between the keypoints of 2 images"""
    
    matches = matcher.knnMatch(desc1, desc2, k=2)
    # good_matches = []
    pts_1 = []
    pts_2 = []
    for m, n in matches:
        if m.distance / n.distance <= 0.8: # perform Lowe's ratio

            if ((kp1[m.queryIdx].pt in pts_1) or (kp2[m.trainIdx].pt in pts_2)):
                continue
            else:
                pts_1.append(kp1[m.queryIdx].pt)
                pts_2.append(kp2[m.trainIdx].pt)

    return np.asarray(pts_1), np.asarray(pts_2)

def compute_quadrilateral(img1, img2):
    """The video is taken from a distance and we need to count NOT track every person in the video
    so we can make a quadrilateral of the information from the previous frames and count the people
     in the new area"""
    """This function computes the Homography between 2 images and gives the quadrilateral that has 
    information from the previous images"""
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # I used SIFT features to compute the hoography, any other techniques like SupePoint, ORB, LoFTr can be used
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Used Brute Force Matcher, but we can use FLANN, SuperGlue, etc.
    src_pts, dst_pts = compute_matches(kp1, kp2, des1, des2)

    # Compute homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

    # Get the shape of the second image (which will be the final canvas size)
    # h2, w2, _ = img2.shape
    # print(h2, w2)

    corners_all = np.array([
                    [0, 0],
                    [0, HEIGHT - 1],
                    [WIDTH - 1, HEIGHT - 1],
                    [WIDTH - 1, 0]
                    ], dtype=np.float32)

    # Warp the first image using the homography matrix
    new_corners = cv2.perspectiveTransform(corners_all.reshape(-1, 1, 2), H)
    quad = Polygon(new_corners.reshape(-1, 2))
    # warped_img1 = cv2.warpPerspective(img1, H, (w2, h2))

    return quad

if __name__ == "__main__":
    # Define the path to the images
    directory = './frames/'
    img_names = os.listdir(directory)
    img_names.sort()

    images_path_list = [os.path.join(directory, img_name) for img_name in img_names]

    # Load the YOLO model 
    model = YOLO("yolov10x.pt")

    # People count in first frame
    image1 = cv2.imread(images_path_list[0])
    HEIGHT, WIDTH, _ = image1.shape
    results = model(image1)
    people_count = 0
    if results[0] != None:
        for box in results[0].boxes:
            if box.cls == 0:  # Class 0 is for 'person' for YOLO model
                people_count += 1

    # People count in the rest of the frames
    for i in range(1, len(images_path_list)):
        
        image1 = cv2.imread(images_path_list[i - 1])
        image2 = cv2.imread(images_path_list[i])
        # computes the homography and makes a quadrilateral using it
        quad = compute_quadrilateral(image1, image2)
        results = model(image2)
        if results[0] != None:
            for box in results[0].boxes:
                if box.cls == 0 and not quad.contains(Point(box.xywh[0][0], box.xywh[0][1])):
                                # Make sure the person bounding-box is not in the quadrilateral
                    people_count += 1

        print(f"Frame {i}: {people_count} people")
