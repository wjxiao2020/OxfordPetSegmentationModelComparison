# Image Semantic Segmentation Model Comparison on Oxford-IIIT Pet Dataset

## Problem Description 
Image segmentation is helpful and useful in a variety of tasks dealing with images. Those tasks might be as trivial as the automatic optimization of color when we use our phones to take pictures every day, or as critical as identifying disease in an early stage on a medical image to significantly boost a patient's survival rate. In this project, we are focusing on training and comparing different machine learning models’ ability to segment cats and dogs from all kinds of random backgrounds. Pet segmentation can be difficult due to their largely varied color that can be very similar to the background as well as all kinds of appearances and shapes (that they can curl their flexible body into). In this report, we will demonstrate the training process and the comparisons of the performance of 4 machine learning models.

## Dataset and Exploratory Data Analysis
The [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) contains 7349 images of cats and dogs of 37 breeds, with around 200 images for each breed. Each image has a corresponding breed classification, head bounding box, and a pixel-level trimap segmentation that annotates each pixel to be one of: the pet body, outline of the pet body, and background. The images come in different shapes. By roughly looking at the dimensions of the images, most of their width and height are around 300 ~ 500 pixels. 

