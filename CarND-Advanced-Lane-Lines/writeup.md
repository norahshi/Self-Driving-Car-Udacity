---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
[//]: # (Image References)

[image1]: ./output_images/undistort_output.jpg "Undistorted"
[image2]: ./output_images/undistort_output_2.jpg "Undistorted_test"
[image3]: ./output_images/threshold_binary.jpg "Binary example"
[image4]: ./output_images/warp.jpg "Warp Example"
[image5]: ./output_images/fit_line.jpg "Fit Visual"
[image6]: ./output_images/output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the sections "Compute the camera calibration using chessboard images" and "Apply a distortion correction to raw images" in the ipython notebook

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Using color transforms, gradients or other methods to create a thresholded binary image. Example of a binary image result included.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at section "Use color transforms, gradients, etc., to create a thresholded binary image"). The final solution relies more on color identification and less on gradients (which are not so useful in shadows or changing road conditions). Specifically, I used HLS, Lab, LUV, and HSV color channels as thresholds. 

Here's an example of my output for this step:
![alt text][image3]

#### 3. Performing perspective transform. Example of a transformed image included.

The code for my perspective transform includes a function called `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identifying lane-line pixels and fitting their positions with a polynomial.
The code include `find_lane()` and `blind_search()` functions.

I first took a histogram of the bottom half of the image. With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame. After searching through all the windows, I concatenated the arrays of indices. Then I extract left and right line pixel positions, and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


The radius of curvature may be given in meters assuming the curve of the road follows a circle.To calculate the radius of curvature of the lane, I took all the pixels of left and right lanes from the warped binary threshold image, defined conversions in x and y from pixels space to meters, fit new polynomials to x,y in world space, and calculate the new radii of curvature in meters. 

For the position of the vehicle, I assumed the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset. As with the polynomial fitting, I converted from pixels to meters.

#### 6. Result plotted back down onto the road such that the lane area is identified clearly. Example image included.

I implemented this step in functions `detect_lane()`, `draw_poly()`,and `process_image ()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### The video pipeline performs reasonably well on the entire project video 

I implemented sanity checks to reject unusable results and replace them with a result from prior frames. I used cv2.matchShapes as a means to make sure the final warp polygon is of quality before using. This step makes sure the polygon for your next frame is close to what it is expected to look like and if not I can elect the new polygon and use old polygon instead. In addition, I added exponential smoothing to averaging over N frames, by updating the new polygon image as follows: New = gamma * New + (1-gamma) * Old, gamma=0.5.

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

#### Potential shortcomings with current pipeline

One potential shortcoming would be what would happen the view of the road would change by other objects in the way, for example when vehicle is driving off the highway or other vehicles are changing lanes in the front. The view of the road would change by the street character, and therefore the binary threshold results would be noisy with other objects in front, and the polynomial lane fit algorithm will not be useful in such scenarios.



#### Possible improvements to pipeline

Potential improvement could be use machine learning methods to design algorithm and parameters that can identify objects within the vehicle view, reiterate training on different street conditions, so as to improve the lane detection reliability.
