# Faster RCNN for Human Detection

## Table of Contents:   
     1. Abstract  
     2. Basic Terminologies    
     3. Dataset   
     4. Frame Generation using Main.py   
     5. MAPREDUCER    
     6. CSV file    
     7. Preprocessing   
     8. Test   
     9. Methadology   
     10. Future Work    
     11. Bibliography   

## Abstract   
Human detection has been a widely researched topic in the field of Computer Vision and Artificial Intelligence.   
It has multiple applications like target detection or traffic/crowd control.   
From image processing to deep learning, human detection techniques have improved both in accuracy and response time.    
This paper is an attempt to detect people from images using Faster-RCNN and replicate results of existing research.   
In order to verify advantages of deep learning techniques over HOG with SVM, a HOG-SVM model was built to run the test images on.    
Amongst deep learning approaches, while YOLO has proven to be the fastest, Faster-RCNN has been very accurate and computationally feasible for industrial approach.


## What is Neural Network?  
A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the **human brain**. 
It is a type of machine learning process that uses interconnected nodes or neurons in a layered structure that resembles the human brain. 
It creates an adaptive system that computers use to learn from their mistakes and improve continuously.


## Convolution Neural Network (CNN)   
CNN is type of Neural network that is designed to process input data that has grid-like structure such as image. It is commonly used in **computer vision**.   
They have three main types of layers, which are:-    
   * Convolutional layer
   * Pooling layer
   * Fully-connected (FC) layer

**Convolutional Layer** applies filters to the input image to extract features such as edges, textures, and shapes.   
The output of Convolutional layer is then passed to Pooling Layer.    

**Pooling Layer** is used to down-sample the image to reduce the spatial dimensions (computations) while retaining the most important information.   

**Fully Connected Layer** is used to make a prediction or classify the image.   

![example](https://github.com/Sazad123/SazRep/assets/119056368/e96bb891-d759-4b02-8129-89292b168dd6)

CNNs are trained using a large dataset of labeled images, where the network learns to recognize patterns and features that are associated with specific objects or classes. Once trained, a CNN can be used to classify new images, or extract features for use in other applications such as object detection or image segmentation.   
# Getting Started with R-CNN, Fast R-CNN, and Faster R-CNN    
Object detection is the process of finding and classifying objects in an image. One deep learning approach, regions with convolutional neural networks (R-CNN), combines rectangular region proposals with convolutional neural network features.  
R-CNN is a two-stage detection algorithm. The first stage identifies a subset of regions in an image that might contain an object. The second stage classifies the object in each region.  

## Object Detection Using R-CNN Algorithms  
Models for object detection using regions with CNNs are based on the following three processes:  
   * Find regions in the image that might contain an object. These regions are called region proposals.  
   * Extract CNN features from the region proposals.
   * Classify the objects using the extracted features.

There are three variants of an R-CNN. Each variant attempts to optimize, speed up, or enhance the results of one or more of these processes.   
**1) R-CNN**   
The R-CNN detector first generates region proposals using an algorithm such as Edge Boxes. The proposal regions are cropped out of the image and resized. Then, the CNN classifies the cropped and resized regions. Finally, the region proposal bounding boxes are refined by a support vector machine (SVM) that is trained using CNN features.  

**2) Fast R-CNN**  
Fast R-CNN uses an algorithm like Edge Boxes to generate region proposals. Unlike the R-CNN detector, which crops and resizes region proposals, the Fast R-CNN detector processes the entire image. Whereas an R-CNN detector must classify each region, Fast R-CNN pools CNN features corresponding to each region proposal. Fast R-CNN is more efficient than R-CNN, because in the Fast R-CNN detector, the computations for overlapping regions are shared.  

**3) Faster R-CNN**  
The Faster R-CNN detector adds a region proposal network (RPN) to generate region proposals directly in the network instead of using an external algorithm like Edge Boxes. The RPN uses Anchor Boxes for Object Detection. Generating region proposals in the network is faster and better tuned to your data. 
![RCNN_eg](https://github.com/Sazad123/SazRep/assets/119056368/d4c26cff-dc1f-4d42-af47-758cbe0b3056)      

## Dataset  
The COCO dataset is used for training and testing. The COCO (Common Objects in Context) dataset is a large-scale image recognition dataset for object detection, segmentation, and captioning tasks.   
It contains over 330,000 images, each annotated with 80 object categories and 5 captions describing the scene. The COCO dataset is widely used in computer vision research and has been used to   
train and evaluate many state-of-the-art object detection and segmentation models.  
The dataset has two main parts: the images and their annotations.  
     * The images are organized into a hierarchy of directories, with the top-level directory containing subdirectories for the train, validation, and test sets.  
     * The annotations are provided in JSON format, with each file corresponding to a single image.  

The below image represents a complete list of 80 classes that COCO has to offer.  
![6666](https://github.com/Sazad123/Group4_RCNN-Sazad-/assets/119056368/405621c1-4a56-4f76-bc9f-10d55db942f1)    

The COCO dataset can be used to train object detection models. The dataset provides bounding box coordinates for 80 different types of objects, which can be used to train models to detect bounding boxes and classify objects in the images.  
![sss](https://github.com/Sazad123/Group4_RCNN-Sazad-/assets/119056368/b2381be4-2eb6-4d6b-96e8-e8e9f4c442f1)


## Main.py  
The Main.py file reads **Crowd.mp4** video file and generates the frames from it. These frames are saved as JPEG file.    

![111](https://github.com/Sazad123/Group4_RCNN-Sazad-/assets/119056368/0930f2bd-a9bb-45c1-9c64-2b6a122cc176)       


Frames (which are in BGR format) are converted into grayscale format.         

![111](https://github.com/Sazad123/Group4_RCNN-Sazad-/assets/119056368/ef1439cd-7dc2-458c-972e-549b8c14bf73)      
BGR frames are saved in **Research>ExtFrame** folder. Grayscale frames are saved in **Research>GRFrame**    

## mapreducer.py   
It is a simple MapReduce program which counts the frequency of words from a sentence or para.  
In this file, we have defined 3 functions:     
     1. mapper()   
     2. reducer()     
     3. main()   

## output.csv   
This file has 7 comma- separated values:    
        | frame-no | object_no | Xmin | Ymin | Xmax | Ymax | ID |     
This csv file is used to store bounding box (x,y) coordinate- values. For each frames, the (Xmin, Ymin) & (Xmax,Ymax) is computed and stored.   
Each frame is allocated with a specific id.  

## Preprocessing  
The preprocessing.py file is used for preprocessing the sample data frames. In this file, the path to the dataset is defined. Then the transforms to be applied to the images and corresponding datasets are defined.
We have defined 3 functions:      
     1. **extract_boxes_labels()**  - Used to extract the bounding boxes and labels from the annotations.   
     2. **get_image_annotation()** - Used to get the image and its annotations.      
     3. **get_preprocessed_data()** - This is the main function to get the preprocessed data to train.   

## store_result.py
This file is used to store the output result in the CSV file after preprocessing. Output is a list which contains a table consisting of:       
     | frame-no | object_no | Xmin | Ymin | Xmax | Ymax | ID |         
We created function: **store_result()**. This function writes the output result in the **output.csv** file when it is in append mode (**a**). The function reads the csv file line by line when it is in read mode(**r**).  

## test.py  
This file is used to test the dataset whether our model works properly or not. This model is defined and evaluated as per need.  
Here we defined a function **detect_human()**. This function takes a list of images, passes to the model and returns the dictionary of predicted objects.   
This function reads the input images & transforms it to a tensor. Then the image is passed to the model for further evaluation.  
For testing purpose, we used class names of **COCO datasets**. The function is further tested.   
Bounding boxes, class labels & prediction scores are separated.   

## Methadology  

#### Faster-RCNN
```
● Take an input image and pass it to the ConvNet which returns feature maps for the image
● Apply Region-Based Convolutional Neural Network (RCNN) on these feature maps and get object proposals
● Apply ROI pooling layer to bring down all the proposals to the same size
● Finally, pass these proposals to a fully connected layer in order to classify any predict the bounding
boxes for the image
```
#### Non-max suppression
In-order to deal with overlapping bounding boxes, non-max suppression is applied.

## Future work
```
● A few of the training images had more than one person. Due to the annotation limitations, only
one bounding box coordinates could be provided to the model. Since the model treats all the
data outside the bounding box as negative samples, there might be the case that a few images of
people were assumed to be negative sample. Such samples might have affected the accuracy of
the model.
● I recently came across a paper ( Martinson, E., and V. Yalla. "Real-time human detection for robots 
using CNN with a feature-based layered pre-filter." _Robot and Human Interactive Communication (RO-MAN), 
2016 25th IEEE International Symposiumon_ . IEEE, 2016.) which provides the neural net training images 
after applying HOG on them to extract relevant features. I would like to try such preprocessing to my model.
```

## Bibliography  

```
● Code References:
  ○ https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch/notebook
  ○ https://pytorch.org/vision/0.9/models.html?highlight=rcnn#torchvision.models.detection.fasterrcnn_resnet50_fpn
● Literature Review
  ○ Girshick, Ross. "Fast r-cnn." ​ Proceedings of the IEEE international conference on computer vision ​. 2015.
  ○ Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." ​ Proceedings of the
  IEEE conference on computer vision and pattern recognition ​. 2016.
  ○ He, Kaiming, et al. "Deep residual learning for image recognition." ​ Proceedings of the IEEE conference
  on computer vision and pattern recognition ​. 2016.
  ○ https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f
  ○ https://towardsdatascience.com/fasterrcnn-explained-part-1-with-code-599c16568cff
  ○ https://medium.com/@madhawavidanapathirana/https-medium-com-madhawavidanapathirana-real-time-human-detection-in-computer-vision-part-1-2acb851f4e
  ○ https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c

● Datasets
  ○ https://cocodataset.org/#home
```
