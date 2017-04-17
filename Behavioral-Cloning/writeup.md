#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



###Model Architecture and Training Strategy

####1. Collecting appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:
- Center lane driving to capture good driving behavior
- Recovering from the left and right sides of the road, so that the vehicle would learn how to get back on track in case it deviated before
- Driving counter-clockwise to help model generalize
- Repeating the above process to get more data points 

After the collection process, I had 27,788 number of data points.

###2. Data Pre-processing

The data preprocessing steps include cropping image, resizing image to 64x64, convert RGB to YUV, normalizing the image data, augmentation of training and validation dataset by flipping the images and angle measurements

- Image normalization, resizing, and color conversion helps model to achieve higher accuracy
- Each image is cropped to focus on only the portion of the image that is useful for predicting a steering angle
- Flipping images and taking the opposite sign of the steering measurement augments the training data sets, and helps model avoid left turn bias involves
- 70% of the images with less than 0.2 steering angle values are randomly removed, because they do not bring additional important information to the training process

I finally put 20% of the data into a validation set. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. NVIDIA Architecture model architecture has been employed. The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description					| 
|:---------------------:|:---------------------------------:| 
| Input         		| 64x64x3 YUV image   				| 
| Convolution 5x5     	| 2x2 stride						|
| RELU					|									|
| Convolution 5x5	    | 2x2 stride						|
| RELU					|									|
| Convolution 5x5	    | 2x2 stride						|
| RELU					|									|
| Convolution 3x3	    | 1x1 stride						|
| RELU					|									|
| Convolution 3x3	    | 1x1 stride						|
| RELU					|									|
| FLatten				| outputs 400						|
| Fully connected		| Output 100						|
| Fully connected		| Output 50							|
| Fully connected		| Output 10							|
| Fully connected		| Output 1							|

####5. Check model overfitting 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The training loss is 0.0302, and validation loss is 0.0327, indicating that there's no significant overfitting issue. I used 2 epochs of training as the training/validation loss did not decrease significantly after the first epoch. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


