#**Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Identify lane lines on the road, first in an image, and later in a video stream.
* Reflect on the work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

[image2]: ./examples/edges.jpg "Edges"

[image3]: ./examples/masked_edges.jpg "Masked Edges"

[image4]: ./examples/lines.jpg "Lines"

[image5]: ./examples/final.jpg "Final"

---

### Reflection

###1. Pipeline Description

My pipeline consisted of 5 steps. 

First, I converted the images to grayscale, with Gaussian smoothing, which is essentially a way of suppressing noise and spurious gradients by averaging.

![alt text][image1]

After that, I used Canny edge detection.

![alt text][image2]

I also created a masked edges image to filter out edges within the region of interest

![alt text][image3]

The fourth step involves running Hough transform and identify lines from edges over certain quality threshold. 

![alt text][image4]

Lastly, I applied the functions on the highway image, and then overlay the detected lanes with original image for visualization output. 

![alt text][image5]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by separating lines into two groups by slope after the initial Hough transform. If the slope is negative, it's upward line and therefore left side lane; similarly if the slope is positive then the line is a right side lane. After that I used numpy.polyfit() function to find the best fitting single line for the two sides, and use that line as the extrapolated left/right lane. 


If you'd like to include images to show how the pipeline works, here is how to include an image: 


###2. Potential shortcomings with current pipeline


One potential shortcoming would be what would happen when vehicle is driving off the highway. The view of the road would change by the street character, and therefore the region of interest parameters we definied would vary based on the street condition. 

Another shortcoming could be edge detection parameters does not apply to rainy/foggy days, because the sight will be more blurred then.

###3. Possible improvements to pipeline

A possible improvement would be to redefine region of interest parameters based on images taken on different roads, and test the results. In that way, we will consider a more reasonable constraint that take into consideration of situations where lanes will apprear in different location of the image, such as when vehicle is turning at the street corner.

Another potential improvement could be use machine learning methods to design algorithm and parameters that can reiterate training on different street and weather conditions, so as to improve the lane detection accuracy. 
