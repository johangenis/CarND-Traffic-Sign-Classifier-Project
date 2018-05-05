# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./barChart.png "Visualization"
[image2]: ./TestAndValidationAccuracyGraph.png "Test and Validation Set Accuracy"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./11_Right-of-way-at-the-next-intersection.jpg "Traffic Sign 1"
[image5]: ./13_Yield.jpg "Traffic Sign 2"
[image6]: ./14_Stop.jpg "Traffic Sign 3"
[image7]: ./25_Road-work.jpg "Traffic Sign 4"
[image8]: ./34_Turn-left-ahead.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### 1. README that including all the rubric points and how you  each one was addressed.The submission includes the project code, and here is a link to my [project code](https://github.com/johangenis/CarND-Traffic-Sign-Classifier-Project/upload/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. A basic summary of the data set. I used the pandas library to calculate summary statistics of the trafficsigns data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32, 32, 3)
* The number of unique classes/labels in the data set is: 43

#### 2. Here is an exploratory visualization of the data set. It is a bar chart showing how the data is classified into 43 classes per set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. The only Pre-processing done, was to normalize the image data, since the results were satisfactory.

#### 2. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs = 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, 10x10x16 |
| RELU     |            |
| Max pooling	      	| 2x2 stride,  outputs = 5x5x16 				|
| Flatten  | Output = 400 |
| Fully connected		| Output = 120          									|
| RELU     |            |
| Fully connected		| Output = 84          									|
| RELU     |            |
| Fully connected		| Output = 10          									|
| Softmax				| Softmax_cross_entropy_with_logits        									|
 

#### 3. To train the model, I used an AdamOptimizer Function, 50 Epochs, a Learning Rate of 0.001 and a Batch Size of 128.

#### 4. My solution used the well known LeNet architecture, since LeNet seemed to be a good fit for classifying traffic signs, because it performed well in class the MNIST data set.

#### My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.947 
* test set accuracy of 0.947

![alt text][image2]
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images were not difficult to classify by the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way at next intersection      		| Right of way at next intersection   									| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| Road Work	      		| Road Work					 				|
| Turn Left Ahead			| Turn Left Ahead      							|

Prediction   Ground Truth
----------   ------------
    11            11     
    13            13     
    14            14     
    25            25     
    34            34     


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


