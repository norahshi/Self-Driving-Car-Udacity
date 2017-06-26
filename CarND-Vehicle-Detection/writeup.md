**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_example.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./examples/bboxes_and_heat.jpg
[video1]: ./project_video.mp4

#### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

### Image Feature extraction 

#### 1. Process of extracting histogram of Oriented Gradients (HOG) features from the training images

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The code for HOG feature extraction is contained in the first code cell of section 'Feature Extraction' in the IPython notebook 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of using HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:

![alt text][image2]

#### 2. Feature parameters selection 

I searched on using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

#### 3. Proceess of training linear SVM includes:

1. making sure I have balanced dataset aka. roughly the same number of cases of each class
2. random Shuffling of the data
3. splitting the data into a training and testing set
4. normalization of features to zero mean and unit variance. Eventually the model reached 99% test accuracy. 

### Sliding Window Search

#### 1. Sliding window search scales and overlap windows parameters 

The sliding window sizes are based on estimates from the test images, and overlap parameters are optimized to find all vehicle images with minimum search efforts. The preliminary model searches all windows and extract hog features for each new window, later on I adopted the HOG Sub-sampling Window Search as a more efficient method of window searching. 

#### 2. Positive detection selection and falst positive removal pipeline

For each vehicle and image, thera are multiple positive detections detected. I recorded the positions of these positive detections. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The heat-map from these detections are essentially used to combine overlapping detections and remove false positives.

Here is an example image including positive detection, heatmap, and final bounding box:

![alt text][image3]

---

### Video Implementation

Here's a [link to my video result](./project_video.mp4). The pipeline performs reasonably well on the entire project video.

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I noticed when vehicles first enter the image there is a lag in terms of positive detection. I could implement with more training images that shows vehicles only partially. 

In addition, there are still occassional false positives showing up in the video output, regardless of relative high threshold of heatmap parameters (currently zeros out pixels overlaps <=2). Potential solution could be multiple lane detections, and exclude all false positive boxes that shows up outside of those lanes. 

