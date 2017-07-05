**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image6]: ./my-signs/speed60kmh.png "Maximum Speed 60 km/h"
[image7]: ./my-signs/stop.png "Stop"
[image4]: ./my-signs/curveleft.png "Dangerous Left Curve"
[image5]: ./my-signs/nopassing.png "No Passing"
[image8]: ./my-signs/yield.png "Yield"

** Rubric Points
***Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
***Writeup / README

****1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bertcuda/carnd-project2/blob/master/Traffic_Sign_Classifier_Restart_09a.ipynb)

***Data Set Summary & Exploration

****1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (34799, 32, 32, 3)
* The size of the validation set is 4410 (4410, 32, 32, 3)
* The size of test set is 12630 (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3) (34799, 32, 32, 3)
* The number of unique classes/labels in the data set is 43 (34799,)

****2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a set of histograms representing the count of examples of each label value, one histogram for each  of training set, validation set and test set.

In addition, a random set of images from the training set is displayed.

[data-visualization]: [./data_visualization.png] "Data Visualization"

***Design and Test a Model Architecture

****1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to improve accuracy. In a paper published in the 2016 IEEE Sixth International Conference on Communications and Electronics (ICCE), [Using grayscale images for object recognition with convolutional-recursive neural network](http://ieeexplore.ieee.org/document/7562656/?part=1), the researchers found that "Experimental results showed that classification with grayscale images resulted in higher accuracy classification than with RGB images across the different types of classifiers."

Here is an example set of traffic sign images after grayscaling.

![After grayscaling][./after_grayscaling.png]

After grayscaling, I normalized the images for brightness and contrast using cv2.equalizeHist() to better distinguish image features. This normalizes the pixel histogram such that the sum of all histogram bins is 255.

Here is an example set of traffic sign images after brightness and contrast.

![After grayscaling][./after_brightandcontrast.png]

As a last step, I normalized the image data to ensure that data points are approximately of the same scale. Otherwise, the numerous multiplications by the weight values would cause data points to diverge excessively. This is not strictly necessary for image data (which ranges from 0 to 255), but is more of a fine-tuning for a small improvement.

I decided not to generate additional data because I achieved satisfactory results with the original data set. However, there exists some opportunity for improvement with augmented data because the data histograms show that label frequency ranges from about 200 to 2000, and therefore some signs are significantly under-represented while others are significantly over-represented.

****2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 400 inputs, 172 outputs        									|
| Dropout		| keep probability of 0.75        									|
| Fully connected		| 86 outputs (nLabels * 2)        									|
| Dropout		| keep probability of 0.75        									|
| Activation		| 43 outputs (nLabels)        									|
| Softmax				| Softmax Cross Entropy with Logits        									|
| Loss				| Reduce Mean        									|
| Optimizer				| Adam Optimizer        									|
|						|												|
|						|												|



****3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
Batch size: 128
Keep probability for dropout: 0.75
Epochs: 15
Learning rate: 0.001

****4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with the sample LeNet architecture, which has been shown to be effective for character recognition and classification, a problem similar to traffic sign classification.

The first run resulted in an accuracy of around 85%. Then, learning that grayscale and normalization preprocessing would improve my results, I implemented those transformations.

My next improvements resulted from changing the network architecture by increasing the width of the first convolutional layer from 6 to 16, and changing the widths of the fully connected layers #3 and #4 to be multiples of the label count.

There was a difference of several percent, with the training and validation results higher than the test results, and so overfitting seemed to be present. Therefore, I added dropout with a "keep probability" of 0.75 after the first two fully connected layers. This reduced the difference but overfitting still seems to be an issue in the final solution.

Finally, I added another preprocessing step for normalizing the brightness and contrast of the input images to better highlight characteristic features in the images.

My final model results were:
* training set accuracy of 98.2%%
* validation set accuracy of 95.5%
* test set accuracy of 91.5%

In addition to a satisfactory overall accuracy level, the probabilities associated with the classified matches are near 100% for 4 correctly-classified traffic signs out of 5 examples from the Web.

***Test a Model on New Images

****1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[image6]: ./my-signs/speed60kmh.png "3: Speed limit 60 km/h"
This sign contains a number, which should be the hardest feature to classify, given that there is no complete training of numbers in the data set.

[image7]: ./my-signs/stop.png "14: Stop"
This sign has a unique hexagon shape and the word "Stop", which should make it more easily classifiable. However, I think the hexagon shape may be interpreted as a circle as well.

[image4]: ./my-signs/curveleft.png "19: Dangerous curve to the left"
This sign contains a left arrow shape, which is similar to several other signs with various configurations of arrows.

[image5]: ./my-signs/nopassing.png "9: No passing"
This sign has two rather small images of cars which could be confused with circles, such as in the "Traffic signals" sign.

[image8]: ./my-signs/yield.png "13: Yield"
This is a simple and clear triangle which I would imagin should be easy to classify, given that all the other signs have additional features within the overall shape.

****2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit 60 km/h					| 3: Speed limit 60 km/h											|
| Stop	      		| 14: Stop					 				|
| Dangerous Left Curve      		| 26: Traffic Signals   									|
| No Passing     			| 9: No Passing 										|
| Yield			| 13: Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. When retested in multiple runs with a freshly-built model in each trial, success rates of 80% to 100% are achieved.

****3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cell 21 of the Ipython notebook. The prediction probabilities are in cell 25.

The softmax prediction probabilities are amazingly high for the correctly-classified signs. This is surprising given the lower test set accuracy percentage of 91.5%.

The sign for "Dangerous curve to the left" was incorrectly classified as "26: Traffic signals" and has a very low prediction probability of 64%, as would be expected for an incorrect result.

For the 4 correctly-classified signs, the probabilities for the alternative predictions are effectively 0%, except for the incorrect classification "Speed limit 80 km/h", which had a probability of 0.13%.

Top prediction:
3 0.99862933158874511719  Speed limit 60 km/h
Other predictions:
5 0.00136944768019020557  Speed limit 80 km/h
8 0.00000092454422429000  Speed limit 120 km/h
6 0.00000014282230154095  End of speed limit 80 km/h
31 0.00000014064157483062 Wild animals crossing

Top prediction:
14 0.99992680549621582031 Stop
Other predictions:
33 0.00005381230948842131 Turn right ahead
4 0.00001772510404407512  Speed limit 70 km/h
17 0.00000151605274822941 No entry
13 0.00000005938898439695 Yield

Top prediction:
26 0.64463102817535400391 Traffic Signals
Other predictions:
19 0.30292659997940063477 Dangerous curve to the left
29 0.04357940331101417542 Bicycles crossing
25 0.00477989437058568001 Road work
4 0.00333137135021388531  Speed limit 70 km/h

Top prediction:
9 0.99929738044738769531  No passing
Other predictions:
38 0.00067734171170741320 Keep right
20 0.00001098814482247690 Dangerous curve to the right
41 0.00000463521928395494 End of no passing
35 0.00000434934736404102 Ahead only

Top prediction:
13 1.00000000000000000000 Yield
Other predictions:
35 0.00000000000201751333 Ahead only
12 0.00000000000000009540 Priority road
9 0.00000000000000003876  No passing
33 0.00000000000000002735 Turn right ahead

*** (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
****1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
