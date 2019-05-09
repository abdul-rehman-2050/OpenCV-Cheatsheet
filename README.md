# OpenCV-Cheatsheet
Sample code and examples for OpenCV, Python 

* [Python](#Python)
* [Pre processing](#Pre-processing)
* [Post processing](#Post-processing)
* [Examples](#Examples)

# Python

## Basic Operations

### Checking the version

```python

import cv2
print cv2.__version__
  
```

### Loading Image in Grayscale

```python
 import cv2
 # Load an color image in grayscale
 img = cv2.imread('messi5.jpg',0)
  
```

### Resize Image

```python
  small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
  #and this will resize the image to have 100 cols (width) and 50 rows (height):

  resized_image = cv2.resize(image, (100, 50)) 

```

### Display Image

```python
 cv2.imshow('image',img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
```

### Check keypress and write Image

```python 
import numpy as np
import cv2

img = cv2.imread('messi5.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0) & 0xFF
if k == 27: # wait for ESC key to exit
  cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
  cv2.imwrite('messigray.png',img)
  cv2.destroyAllWindows()

```

### Using Matplotlib

Its always handy to display images using matplotlib library. So here is an example of doing this 

```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
plt.show()

```

# Pre-processing 

### Grayscale to binary Threshold method

```python
  
  ret,thresh = cv2.threshold(frame_threshed,127,255,0)
  
```

### Median Blur Filter
```python
  img = cv2.medianBlur(img,5)

```

### Gaussian Filter

```python

blur = cv2.GaussianBlur(img,(5,5),0)

```

### Otsu's Thresholding

```python
  # Otsu's thresholding
  ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```


### Erosion

```python
import cv2
import numpy as np

img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

```

### Cuntour Detection

```python
  
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

```

# Post-processing

[Display Text on Image](https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2)

```python
  cv2.putText(image,"Hello World!!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
```
[Crop Image with x,y,w,h](https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)

```python

import cv2
img = cv2.imread("lenna.png")
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

```


# Examples

### [Find Biggest Color Object OpenCV-2.4](https://stackoverflow.com/questions/16538774/dealing-with-contours-and-bounding-rectangle-in-opencv-2-4-python-2-7)

```python 

import numpy as np
import cv2

im = cv2.imread('shot.bmp')
hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
COLOR_MIN = np.array([20, 80, 80],np.uint8)
COLOR_MAX = np.array([40, 255, 255],np.uint8)
frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
imgray = frame_threshed
ret,thresh = cv2.threshold(frame_threshed,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# Find the index of the largest contour
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("Show",im)
cv2.waitKey()
cv2.destroyAllWindows()


```

### Capture Live Webcam Feed

```python

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  ret, frame = cap.read()
  
  # Our operations on the frame come here
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Display the resulting frame
  cv2.imshow('frame',gray)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

```

### Playing video from file

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```
