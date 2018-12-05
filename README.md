# OpenCV-Cheatsheet
Sample code and examples for OpenCV, Python 


# Python

## Basic Operations

* Checking the version

```python

import cv2
print cv2.__version__
  
```

* Loading Image in Grayscale

```python
 import cv2
 # Load an color image in grayscale
 img = cv2.imread('messi5.jpg',0)
  
```

* Display Image

```python
 cv2.imshow('image',img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
```

* Check keypress and write Image

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

* Using Matplotlib

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

## Pre-processing 

## Post-Processing

* [Display Text on Image](https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2)

```python
  cv2.putText(image,"Hello World!!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
```


## Examples

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
