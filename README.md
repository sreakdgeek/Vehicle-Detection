## Vehicle Detection

### Introduction

Vehicle Detection project intends to build a pipeline that detects vehicle(s) in a video stream.
This task can be aporached in multiple ways using an end-to-end deep learning architecture or 
a classification pipeline using hand-engineered features. Classical Computer Vision technique
requires extracting features such as HOG (Histogram of Gradients), color histograms, spatial bins, etc.

Alternatively, deep learning techniques such as YOLO (You only Look Once) or SSD (Single Shot Detector)
can be applied. This implmentation is based on classical computer vision techniques.  

In contrast to a classification problem, this is a detection problem. In such a case we may have
one or more classes of objects may appear one more times in an image. In classical approach, instead of
passing the entire image, we slide across the window and pass the image patches to the classifier to
verify if an instance of a class exists in the patch. This requires a training set consists of 
postive and negative class images.

In deep learning approach, a training set consisting of bounding box coordinates along with classes 
are passed to a classifier.

This implementation uses a classical approach of using classifier, hand extracted features and a sliding window detection.

Below are the steps requied typically:

*	Data exploration
*	Feature Engineering
*	HOG Features
*	Color Histogram
*	Spatial Bin Features
*       Classification
*	Sliding window search
*       HOG Sub-sampling	
*       Heatmaps - Rejecting False positives	
*       Pipeline
*	Discussion


[//]: # (Image References) 
[image1]: ./images/car_no_car.PNG
[image2]: ./images/hog_image_viz.PNG
[image3]: ./images/classification_metrics.PNG
[image4]: ./images/sliding_window.PNG
[image5]: ./images/overlap_experimentation.PNG
[image6]: ./images/heat_map1.PNG
[image7]: ./images/heat_map2.PNG
[image8]: ./images/heat_map3.PNG
[image9]: ./images/heat_map4.PNG
[image10]: ./images/Vehicle_Detection_Shapshot.PNG

---
### Data Exploration

Here is a brief look of car and non-car images. Thankfully, this is a balanced data set with nearly 50% of car and no-car images:


![alt text][image1]


### Feature Engineering

From vehicle images we can extract the below features:

1. HOG (Histogram of Gradients)
2. Color Histogram
3. Spatial Bin Features


#### HOG Features

Typically HOG features are extracted using below steps:

1. Global image normalization
2. Compute the image gradients in x and y direction
3. Compute the gradient histograms
4. Normalize across the blocks
5. Flattent the feature vector

Below is HOG image visualization:

![alt text][image2]

As can be seen from the image, HOG features are quite informative detecting the edges while reducing the background noise.
In HOG implementation, histogram of gradients is typically computed in a 8x8 cell. This can be tuned based on the characteristics
of the type of object that we are attempting to detect. The magnitude of gradient within this 8x8 cell can be captured in
a histogram of size say 9 or 18 based on the direction of gradient. Thus each bin size could capture 20 or 10 degrees respectively.
So, magnitude of gradient is captured in a histogram or bins of the orientation. Since the image would be sensitive
to lighting conditions, histogram needs to be normalized. Thiss is achieved by dividing the feature vector by its L-2 Norm.
Vectors from each block are concatenated to form a flattened feature vector.

In my implmentation, I have experimented with 18 bins and found better results.  Below were the HOG descriptor paramters 
used in this implementation:

color_space = 'YCrCb'
orient = 18  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"


#### Color Histogram Features

Color histogram captures the pixel intensities in the given color space such as RGB, YUV, Y2CrCb, etc.
For example, for RGB color space, counts of pixels in a given bin (range of pixel intensities) is captured. Such histogram
is flattened and can be used as a feature vector. Hope is that such a feature vector would be able to distinguish an 
object such as a car and non-car. While cars may share a pattern in color spatial/shape information may not be captured
in a color histogram. 

#### Spatial bin Features

Spatial binning involves resizing the image to a lower resolution and use the raw pixel intensities as feature vector.


Features are further scaled using scaler such as a standard or min-max scaler. In this case, standard scaler has been used which
centers the features with zero mean. After concatenating HOG, color histogram and spatial binned features, 
I have used SelectKBest() from sklearn() to further reduce the feature dimensionality form ~11K to ~8K.

### Classification


Below are the set of classifiers that I have experimented with:

![alt text][image3]


Ensemble of Non-linear SVC, GBM, Random Forests, Gaussian Naive Bayes provided excelent results with an accuracy of ~99.7%.
However, unfortunately the predict() function takes more than a second to predict the class. Also, ensemble performed excellently
with minimal false positives. Given that we would be running the classifier on multiple image patches, slow predict() of
ensemble classifier proved to be undesireable. Thus, Linear SVC was selected as classifier to detect the vehicle.


### Sliding Window Search


Sliding window search involves diving the image into smaller windows and slide across the window and search (classify) for
the vehicle using different scales. Also, areas where vehicle is not expected is removed from the search and thus avoiding
false positives. Below image shows 96x96 window size with an xy-overlap of 50%:

![alt text][image4]

Below shows the various experiments run for a combination of xy-overlaps and start


![alt text][image5]



### HOG Sub-sampling

While sliding window is quite effecting in searching for the object, the operation is quite costly since it involves 
extracting the HOG features for each image patch. Instead of extracting the HOG features for each image patch, 
HOG sub-sampling compute the HOG features for the entire image, and HOG features are sub-sampled by indexing
into the HOG features based on the image patch location.

After experimenting with various start/stop Y-coordinates and scale values, below parameters were
finalized:

param_list = [(400, 600, 1.2), (400, 470, 1.0), (420, 480, 1.0), 
              (400, 500, 1.5), (430, 530, 1.5), (400, 530, 2.0), 
              (470, 660, 3.0)]

### Heat Maps


One of the problems using a simple classifier such as Linear SVC is that there could be multiple false positives.
In order to eliminate false positives, heatmaps were used. The idea of heatmaps is that when classifier runs 
on multiple sliding windows, there is a good chance that multiple true positives could be detected and
there by increasing intensities around the detected objects. Moreover, generating heatmap over last few frames
(say 20), false positives more false positives could be eliminated. Using a threshold value, false detections
could be eliminated. 


Below image shows the initial heat map:

![alt text][image6]


After applying threshold:

![alt text][image7]


Gray-scale image:

![alt text][image8]


Car positions:

![alt text][image9]

### End to End Pipeline

Below are the steps performed for End-to-End pipeline:

1. Build classifier using features such as HOG
2. Sliding-window search / HOG Subsampling
3. Apply heatmaps
4. Smoothing using heatmaps across last 'k' frames
5. Detect and draw bounding box 

Below shows examples of cars detected:

![alt text][image10]

### Video Generation

Below is the ink to the Video generated:


### Discussion


While this approach of using SVM classifier with HOG features was nearly effective, implmentation still generates few false positives and detection could be much more smoother.
Also, more complex classifiers such as an ensemble of multiple classifiers could not be used because of slow predict() function. 

Below are the additional approaches that can be taken to make the implementation more effective:


1) Smoothing function using average of overlappped windows. Overlap of detected windows can be computed to make the detection more stable and robust.
2) Deep Learning approaches such as YOLO, Fast R-CNN or SSD could be used as they have been proven to be more effective and significantly faster in detecting objects of different classes.
3) Further technique such as Kalman Filtering can be used to track the objects as we need not detect the object in every frame. Kalman fitlering allows us to predict the position of object
   to be tracked using its spatial position and velocity.
