# OpenCV-Cheatsheet
Sample code and examples for OpenCV, Python 


# Python

## Checking the version

```python

import cv2
print cv2.__version__
  
```

## Loading Image in Grayscale

```python
 import cv2
 # Load an color image in grayscale
 img = cv2.imread('messi5.jpg',0)
  
```

## Display Image

```python
 cv2.imshow('image',img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
```

## Check keypress and write Image

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


