# OpenCV

## What is OpenCV and why is it used?

OpenCV (Open Source Computer Vision Library) is an open-source library focused on computer vision and image processing. It's used because:
- Provides optimized algorithms for image/video processing
- Supports multiple programming languages (C++, Python, Java)
- Has both CPU and GPU acceleration support
- Offers real-time image processing capabilities
- Contains 2500+ optimized algorithms
- Is free for both academic and commercial use

## What are the core concepts in OpenCV?

The core concepts include:
1. **Images as Arrays**: Images are represented as numpy arrays in Python
2. **Color Spaces**: Different ways to represent color (RGB, BGR, HSV, etc.)
3. **Pixels**: Individual points in an image with intensity values
4. **Channels**: Color components of an image (B, G, R channels)
5. **Contours**: Curves joining continuous points along a boundary
6. **Features**: Interesting parts of an image (corners, edges, etc.)

## How do you read, display, and save images in OpenCV?

```python
import cv2
import numpy as np

# Read an image
img = cv2.imread('image.jpg')  # Returns None if image not found
# Default reading is BGR format
# cv2.IMREAD_COLOR: Default BGR color
# cv2.IMREAD_GRAYSCALE: Grayscale
# cv2.IMREAD_UNCHANGED: Image with alpha channel

# Display image
cv2.imshow('Window Name', img)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()

# Save image
cv2.imwrite('output.jpg', img)
```

## How do you perform basic image operations?

1. **Accessing and Modifying Pixels**:
```python
# Access pixel value
px = img[100, 100]  # Returns [B, G, R] value
# Modify pixel value
img[100, 100] = [255, 255, 255]  # Set to white

# Get image properties
height, width, channels = img.shape
```

2. **Region of Interest (ROI)**:
```python
# Extract region
roi = img[100:200, 100:200]
# Copy region to another location
img[300:400, 300:400] = roi
```

3. **Color Space Conversions**:
```python
# BGR to Gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

## What are the common image processing operations in OpenCV?

1. **Resizing Images**:
```python
# Resize with specific dimensions
resized = cv2.resize(img, (width, height))
# Resize by scale
resized = cv2.resize(img, None, fx=0.5, fy=0.5)
```

2. **Image Filtering**:
```python
# Gaussian Blur
blurred = cv2.GaussianBlur(img, (5,5), 0)
# Median Blur
median = cv2.medianBlur(img, 5)
# Bilateral Filter
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

3. **Thresholding**:
```python
# Simple thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(gray, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
# Otsu's thresholding
ret, otsu = cv2.threshold(gray, 0, 255, 
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## How do you detect edges and contours?

1. **Edge Detection**:
```python
# Canny Edge Detection
edges = cv2.Canny(img, 100, 200)

# Sobel Edge Detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
```

2. **Contour Detection**:
```python
# Find contours
contours, hierarchy = cv2.findContours(thresh, 
                                     cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0,255,0), 3)

# Contour properties
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    # Bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt)
```

## How do you perform feature detection and matching?

1. **Corner Detection**:
```python
# Harris Corner Detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
```

2. **SIFT (Scale-Invariant Feature Transform)**:
```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None)
```

3. **Feature Matching**:
```python
# Brute-Force Matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# FLANN Matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
```

## How do you perform face detection?

Using Haar Cascades:
```python
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around faces
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
```

## How do you handle video in OpenCV?

```python
# Capture video from camera
cap = cv2.VideoCapture(0)  # 0 for default camera

# Capture video from file
cap = cv2.VideoCapture('video.mp4')

# Write video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

# Read frames
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Process frame here
        
        # Write frame
        out.write(frame)
        
        # Display frame
        cv2.imshow('Frame', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
```

## What are some common image transformations?

1. **Geometric Transformations**:
```python
# Translation
M = np.float32([[1,0,100],[0,1,50]])
translated = cv2.warpAffine(img, M, (cols, rows))

# Rotation
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
rotated = cv2.warpAffine(img, M, (cols, rows))

# Scaling
scaled = cv2.resize(img, None, fx=2, fy=2)

# Affine Transform
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
affine = cv2.warpAffine(img, M, (cols,rows))
```

2. **Perspective Transform**:
```python
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
perspective = cv2.warpPerspective(img, M, (300,300))
```

## How do you optimize OpenCV performance?

1. **Memory Management**:
```python
# Release resources
cv2.destroyAllWindows()
cap.release()

# Use grayscale when color isn't needed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

2. **Processing Optimization**:
```python
# Use smaller images when possible
small = cv2.resize(img, None, fx=0.5, fy=0.5)

# Use ROI instead of full image when possible
roi = img[y:y+h, x:x+w]

# Use appropriate data types
img = img.astype(np.uint8)
```

## What are some common image segmentation techniques?

1. **Watershed Segmentation**:
```python
# Marker-based watershed
markers = np.zeros(gray.shape, dtype=np.int32)
markers[sure_fg] = 2
markers[sure_bg] = 1
cv2.watershed(img, markers)
```

2. **Grabcut Segmentation**:
```python
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
rect = (50,50,450,290)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
```

## How do you handle color tracking?

```python
# Define range of color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)
```