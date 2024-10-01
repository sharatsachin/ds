# Computer Vision

## What is image convolution and why is it important in computer vision?

Image convolution is the process of applying a filter (kernel) to an image to create a feature map. Mathematically:

$(I * K)(i,j) = \sum_{m}\sum_{n} I(i-m,j-n)K(m,n)$

Where I is the image and K is the kernel.

It's important because it:
1. Detects features (edges, textures)
2. Reduces noise
3. Forms the basis of CNNs

## Explain the architecture of a basic Convolutional Neural Network (CNN).

A basic CNN architecture:
1. Input layer: Raw image pixels
2. Convolutional layers: Apply filters to detect features
3. Activation function (e.g., ReLU): Introduces non-linearity
4. Pooling layers: Reduce spatial dimensions
5. Fully connected layers: Combine features for classification
6. Output layer: Final classification/regression

## What is the difference between object detection and image segmentation?

Object detection: Identifies and locates objects in an image, typically with bounding boxes.

Image segmentation: Assigns a class label to each pixel in the image. Two main types:
1. Semantic segmentation: Labels pixels by class, doesn't distinguish instances
2. Instance segmentation: Labels pixels by class and distinguishes individual instances

## How does transfer learning work in the context of image classification?

Transfer learning in image classification:
1. Start with a pre-trained model (e.g., VGG, ResNet) on a large dataset
2. Remove the last fully connected layer(s)
3. Add new layer(s) for the target task
4. Fine-tune: either train only the new layers or fine-tune the entire network with a small learning rate

This leverages general features learned from large datasets to improve performance on smaller, specific datasets.

## What is data augmentation and why is it crucial for computer vision tasks?

Data augmentation artificially increases the training set by applying transformations to existing images. Common techniques:
1. Rotation, flipping, scaling
2. Color jittering
3. Random cropping
4. Adding noise

It's crucial because it:
1. Increases dataset size without collecting new data
2. Improves model generalization
3. Reduces overfitting
4. Helps models learn invariance to certain transformations

## Explain the concept of feature maps in CNNs.

Feature maps are the outputs of convolutional layers. Each feature map represents the response of a specific filter applied to the input. Early layers typically detect low-level features (edges, textures), while deeper layers detect more complex, high-level features.

## What is the role of pooling layers in CNNs?

Pooling layers:
1. Reduce spatial dimensions of feature maps
2. Decrease computational complexity
3. Introduce spatial invariance
4. Help prevent overfitting

Common types:
- Max pooling: Takes maximum value in each window
- Average pooling: Takes average value in each window

## How does the YOLO (You Only Look Once) algorithm work for object detection?

YOLO:
1. Divides image into a grid
2. For each grid cell, predicts:
   - Bounding boxes and confidence scores
   - Class probabilities
3. Uses a single neural network for the entire image
4. Performs detection in one forward pass, hence "You Only Look Once"
5. Combines bounding box predictions and class probabilities
6. Applies non-max suppression for final detections

YOLO is faster than two-stage detectors but may struggle with small objects.

## What is the difference between R-CNN, Fast R-CNN, and Faster R-CNN?

1. R-CNN:
   - Proposes regions, extracts features with CNN, classifies with SVM
   - Slow, processes each region separately

2. Fast R-CNN:
   - Uses RoI pooling to extract features from proposed regions
   - Single CNN pass for the whole image
   - Faster than R-CNN

3. Faster R-CNN:
   - Introduces Region Proposal Network (RPN)
   - End-to-end trainable
   - Significantly faster than Fast R-CNN

## Explain the concept of anchor boxes in object detection.

Anchor boxes are predefined bounding boxes of various scales and aspect ratios. They serve as reference boxes for object detection algorithms. Key points:

1. Used in algorithms like YOLO and Faster R-CNN
2. Help detect objects of different sizes and shapes
3. Network predicts offsets from these anchor boxes
4. Improve detection of objects with varying aspect ratios

## What is the Intersection over Union (IoU) metric and how is it used?

IoU measures the overlap between predicted and ground truth bounding boxes:

$IoU = \frac{Area of Overlap}{Area of Union}$

Uses:
1. Evaluating object detection performance
2. Non-max suppression in object detection
3. Defining positive/negative examples in training
4. Threshold for considering a detection correct (e.g., IoU > 0.5)

## How does a Generative Adversarial Network (GAN) work for image generation?

GANs consist of two competing neural networks:
1. Generator (G): Creates fake images
2. Discriminator (D): Distinguishes real from fake images

Training process:
1. G generates fake images
2. D tries to distinguish real from fake
3. G aims to fool D
4. Networks improve through adversarial training

Objective function:
$\min_G \max_D V(D,G) = E_{x\sim p_{data}(x)}[\log D(x)] + E_{z\sim p_z(z)}[\log(1-D(G(z)))]$

## What is the difference between supervised and unsupervised learning in computer vision?

Supervised learning:
- Uses labeled data
- Examples: Image classification, object detection
- Learns mapping from input to known output

Unsupervised learning:
- Uses unlabeled data
- Examples: Clustering, dimensionality reduction, anomaly detection
- Finds patterns or structure in data without explicit labels

## Explain the concept of image segmentation and list a few popular algorithms.

Image segmentation divides an image into multiple segments or objects. It assigns a label to every pixel.

Popular algorithms:
1. Fully Convolutional Networks (FCN)
2. U-Net
3. Mask R-CNN
4. DeepLab
5. SegNet
6. PSPNet (Pyramid Scene Parsing Network)

## What is the purpose of non-max suppression in object detection?

Non-max suppression (NMS) reduces multiple detections of the same object to a single detection. Process:

1. Sort detections by confidence score
2. Keep highest scoring detection
3. Remove detections with high IoU with kept detection
4. Repeat steps 2-3 for remaining detections

Purpose:
- Reduces redundant detections
- Improves precision of object detection
- Simplifies final output

## How does facial recognition work? Explain the basic steps involved.

Basic steps in facial recognition:

1. Face detection: Locate faces in the image
2. Face alignment: Normalize face position, size, and pose
3. Feature extraction: Extract distinctive facial features
4. Feature matching: Compare extracted features with database
5. Decision making: Determine identity based on similarity score

Common techniques:
- CNNs for feature extraction
- Siamese networks for face comparison
- Triplet loss for learning discriminative features

## What are some common preprocessing techniques used in computer vision?

Common preprocessing techniques:

1. Resizing: Standardize image dimensions
2. Normalization: Scale pixel values (e.g., to [0,1] or [-1,1])
3. Color space conversion (e.g., RGB to grayscale)
4. Histogram equalization: Enhance contrast
5. Noise reduction: Gaussian or median filtering
6. Data augmentation: Flipping, rotation, scaling
7. Channel-wise mean subtraction
8. Gaussian blur for reducing high-frequency noise

## Explain the concept of feature extraction in traditional computer vision vs. deep learning approaches.

Traditional CV feature extraction:
- Manually designed features (e.g., SIFT, SURF, HOG)
- Based on domain knowledge and mathematical properties
- Often interpretable but may not capture all relevant information

Deep learning feature extraction:
- Learned automatically from data
- Hierarchical, from low-level to high-level features
- Can capture complex patterns but less interpretable
- Often more effective for complex tasks

## What is the difference between semantic segmentation and instance segmentation?

Semantic segmentation:
- Assigns class label to each pixel
- Doesn't distinguish between instances of the same class
- Output: Single label map

Instance segmentation:
- Assigns class label and instance ID to each pixel
- Distinguishes between instances of the same class
- Output: Label map with instance IDs

Example: In a street scene, semantic segmentation labels all cars as "car", while instance segmentation uniquely identifies each individual car.

## How does optical flow estimation work and what are its applications?

Optical flow estimates motion between video frames. Basic principle:

$I(x,y,t) = I(x+dx, y+dy, t+dt)$

Where I is image intensity, and (dx,dy) is the displacement.

Methods:
1. Lucas-Kanade: Assumes constant flow in local neighborhood
2. Horn-Schunck: Global smoothness constraint
3. Deep learning approaches: FlowNet, PWC-Net

Applications:
1. Motion estimation in video compression
2. Object tracking
3. Action recognition
4. Video stabilization
5. 3D reconstruction from motion