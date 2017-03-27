# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./category_counts.png "image_summarization1"
[image2]: ./after_augmentation_category_counts.png "image_summarization2"
[image3]: ./before_and_after_augment.jpg "augmentation"
[image4]: ./before_grayscale.jpg "before_grayscale"
[image5]: ./after_grayscale.jpg "after_grayscale"
[image6]: ./traffic-signs-data/additional/1.jpg "1"
[image7]: ./traffic-signs-data/additional/2.jpg "2"
[image8]: ./traffic-signs-data/additional/3.jpg "3"
[image9]: ./traffic-signs-data/additional/4.jpg "4"
[image10]: ./traffic-signs-data/additional/5.jpg "5"





## Rubric Points

---
You're reading it! and here is a link to my [project code](https://github.com/norahshi/Self-Driving-Car-Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic%2BSign%2BClassifier.html)

### Data Set Summary & Exploration

#### 1. The submission includes a basic summary of the data set 
The code for this step is contained in the "Step 1: Dataset Summary & Exploration" section of the ipython notebook.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. The submission include an exploratory visualization of the dataset 

The code for this step is contained in the third code cell of the IPython notebook.  
Here is an exploratory visualization of the data set. It is a bar chart showing how many training data sets there are for each label

![summary1][image1]

### Design and Test a Model Architecture

#### 1. The data preprocessing steps include augmentation of training dataset, resizing all training/validation(already given as converted) and testing data, convert RGB to grayscale, and normalizing the image data

As a first step, I decided to agument the image. One big problem in this data set is that unbalanced number of data for each class, as shown earlier in the image summarization. If one class has significantly more training dataset, the model will be skewed as well. I keep the number of samples for each class the same as 2000 by image augmentation techniques including random rotation or shifting of the traffic sign. 

The new training dataset category count summarization is as follow:
![summary2][image2]

One example of dataset augmentation is as follow:
![augmentation][image3]

Secondly, I converted the images to grayscale because color is not a useful feature in identifying traffic signs.

Here is an example of a traffic sign image before and after grayscaling.
![before][image4] ![after][image5]

As a last step, I normalized the image data so that each feature has a similar range, to keep the gradients in control (and that we only need one global learning rate multiplier)

#### 2. Model type, layers, layer sizes, connectivity

The code for my final model is located in the "training pipeline" section the ipython notebook. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image  						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| FLatten				| outputs 400									|
| Fully connected		| Input 400. Output 120							|
| RELU					|												|
| Dropout 				|												|
| Fully connected		| Input 120. Output 84							|
| RELU					|												|
| Dropout 				|												|
| Fully connected		| Input 84. Output 43							|


#### 4. The code for training the model is located from section "set up tensorflow" to "training pipeline" in the Ipython notebook

To train the model, I used AdamOptimizer algorithm. The hyperparameters chosen are as follow:
EPOCHS = 20
BATCH_SIZE = 128
dropout = 0.7
learning_rate = 0.001
regularization = 0.001

#### 5. Training model
I initially chose the LENET model because it has been used on MNIST dataset with high accuracy, and can solve similar problems of traffic light classification. 

To improve on the LENET model infrastructure, I compare the result of test/validation set. I included the accuracy/loss plot to help visualize the accuracy/loss progress throughout the epochs. 

Here are the steps of adjusting the hyperparameters:
* If the learning rate is too high, the loss curve falls very sharply initially; when learning rate is too low, the training/validation accuracy was low even after 20 epochs. After iterations, I decided on setting the learning rate as 0.001.
* A extreme high accuracy on the training set (99%) but low accuracy on the validation set indicates over fitting. Therefore, I included data augmentation in the training data set preprocessing, and train/test accuracy became more consistent and on the same scale.
* When training and validation loss curve diverge and validation loss increase overtime, it also indicates over fitting. I increased dropout and regularization in the fully connected layers to reduce over fitting. 
* A low accuracy on both training and validation sets indicates under fitting. I lowered dropout and regularization rates.
* When training accuracy steadily increases, while validation accuracy oscillates, the model is overfitting. So I reduced the # of epochs for training.

The code for calculating the accuracy of the model is located in the "train the model" and "evaluate model" sections of the Ipython notebook.

My final model results were:
* training set accuracy of 96%
* validation set accuracy of 93%
* test set accuracy of 90%

### Test a Model on New Images

#### 1. I chose five German traffic signs found on the web, and show as below. Potential common issues for these five images are that they have noisy background image, and also have watermarks on top, which could have cause issue predicting the right label.

![1][image6] 
![2][image7] 
![3][image8] 
![4][image9] 
![5][image10]

#### 2. The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The code for making predictions on my final model is located in the "Step 3. Test a model on new images" of the Ipython notebook. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h       		| 20 km/h   									| 
| Stop sign    			| Stop sign										|
| Roundabout mandatory	| Roundabout mandatory							|
| Priority road    		| Priority road 				 				|
| Pedestrian			| Pedestrian      								|


#### 3. This section shows certainty of model when predicting on each of the five new images, using the softmax probabilities.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.79834139e-01		| 20 km/h   									| 
| 1.53847300e-02   		| 30 km/h										|
| 4.22899704e-03		| 70 km/h										|
| 1.65056248e-04		| General caution				 				|
| 9.86562518e-05	    | End of speed limit (80km/h)					|


For the second image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.90796089e-01		| Stop sign   									| 
| 7.14431470e-03		| Keep right									|
| 8.34851584e-04		| No entry  									|
| 4.62959317e-04		| Yield     					 				|
| 2.19367023e-04	    | Turn right ahead  							|

For the third image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.73165989e-01		| Roundabout mandatory							| 
| 2.04627030e-02		| Go straight or left							|
| 5.95692219e-03		| Keep left 									|
| 2.45149480e-04		| Pedestrians					 				|
| 6.32660813e-05	    | Stop     										|

For the fourth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.98085475e-01		| Priority road 								| 
| 1.00492157e-01		| Roundabout mandatory							|
| 5.13489649e-04		| Keep left 									|
| 4.72170621e-04		| Stop      					 				|
| 1.80467774e-04	    | Turn right ahead     							|

For the fifth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.53833771e-01		| Pedestrians  									| 
| 2.45217115e-01		| Road narrows on the right						|
| 5.37009153e-04		| General caution   							|
| 4.03766986e-04		| Children crossing 			 				|
| 5.21021457e-06	    | Right-of-way at the next intersection			|

