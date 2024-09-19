# Crowd counting Using YOLO-v10 and Homographies
In this project I came up with a method count the number of people in a crowd, the video I used was taken by a drone. For the project to work the video must be taken from a far away distance like the drone video and have a good object detector to detect people. The proposed method takes advantage of the long distance imaging nature of drone videos

## Watch the Demo Video

[![example clip](./clip.mp4)](./clip.mp4)


## Initial setup and Video Understanding

The video is taken from a far away distance using a drone and it is a sequence of overlapping images. All the objects in the video are from a far away distance with overlapping fields between successive images, this can be used to compute a homography.

## Advantages of Using Homography instead of DeepSORT

1. In the video there is a huge amount of crowd and running DeepSORT or any other Tracking algorithm would be computationally very high as it is O(N*M) complexity, where N is number of people already tracked and M is the number of people in the frame.
2. The same problem would appear with any other type of feature extraction based tracker for example YOLO-v10t, MOTR or ByteSORT
3. We could use SIFT instead of DeepSORT to reduce the computational overhead but the complexity would still be the same, and also we don't actually need to track the movements of people in the video, we just need to count the number of people
4. By computing Homography between 2 images we could easily understand what new information is the new image bringing and add only the people in that area. 
5. The constraint for such approach is the objects in frame shouldn't be moving, but in our specific case the frame-rate is high and people moving at a far distance is not as high as in pixel co-ordinates, which is why the specific approach would still work

## Approach followed to count the number of People

For a given sequence of images, I still used an Object detector to detect people in the video, I couldn't come up with a better way to reliably detect people in a given frame

1. The first step is on the first image run an object detector to count the number of people and add this to the people_counter variable
2. From the second image to the final image compute the homography between current image and the previous image, this would help in overlaying the previous image on the current image
3. when the previous image is overlayed on the current image, the new information in the current image can be easily obtained, this would help in adding the people in new information zone to the people_counter
4. To do the above step efficiently I used the corners of each image and transformed it current image using the Homography technique. I used Shapely library to form a quadrilateral of transformed co-ordinates and checking if a person is in that quadrilateral
5. This would yield reliable results given the Object detector works well 

However there is one edge case which I didn't work for after looking at the video, that is loop-closures. If the video is drone moving in a circle then the people would be mapped over and over again. This can be avoided by tracking the co-ordintaes of all previous frames and making a polygon instead of quadrilateral by performing a `union` operation in Shapely. I didn't work on this edge case as the video is moving forward than going back to where it started

## Approach for Homography

To compute the Homography between 2 image I used the standard procedure itself.
1. computed the keypoints and descriptors for both images using feature extractors like ORB, SIFT, BRIEF, LoFTr, SuperPoint
2. Compute the correspondences between two images using a matcher like Brute-force matcher, FLANN matcher or SuperGLue. These correspondences still have a lot of noise in them because of moving elements or repeating elements, to avoid this we perform Lowe's ratio
3. To further remove outliers in the correspondences we perform RANSAC while  computing Homography that helps in finding the inliers as well.

## Files

`video_to_images.py` - This file converts the video to a series of images and puts them in a folder for ease of use

`homography_counter.py` - This file consists the code to count the number of people in the video(series of images from folder) using the above discussed technique
`clip.mp4` - This is the clip from YouTube in mp4 format as per instructions in the question

## My system specifications

I used Ubuntu 20.04 with Miniconda installed. I use a 16 core 8GB RAM laptop. Each frame took about 6 seconds to process and count the number of people for a 1280X720 image

## Instructions to RUN

To set the initial environment variables run the following command
``` shell
conda env create --file environment.yaml
```
The above command creates a anaconda virtual environment with the same variables I used and ran the code
``` shell 
conda activate lauretta
```
The above command starts the virtual environment and run the following commands in this environment
The current code in this repo is built to work for the video in this repository ```clip.mp4```.

Before running the people computation algorithm we need to parse the video to a series of files and place them in a folder. Run this command to perform that task on the given video ```clip.mp4```
``` shell
python video_to_images.py
```
This saves the video in a folder called "frames" and sequentially numbers the individual frames
Now we can Run the actual file algorithm using the following command
``` shell
python homography_counter.py
```
The above command should give the cumilative number of people in the crowd till that frame 

## Other Approaches Considered

1. I considered using the outliers in homography computation to be people but the problem is one person can have multiple keypoints
2. I considered Flow based technique to segment people, but that wouldn't solve the problem of identifying individual people, actually this lead to the idea of using homography
3. I considered using DeepSORT, SIFT and other feature trackers to track people but it would be computationally expensive,specifically for huge crowds such as this one
4. We would face the same type of problems with end-to-end Neural networks like MOTR, TO reduce the computational overhead I used this approach

## Problems Faced

I couldn't get an accurate estimate of the number of people due to detecting people in the video. I used freely available YOLO models to detect people but they weren't able to detect people. This could be due to the lack of training data of people from top-view or YouTube's compression might have changed the video slighly resulting in poor performance of YOLO