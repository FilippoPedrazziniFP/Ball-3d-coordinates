import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

IMAGE_H = 814
IMAGE_W = 1360

#### TOP VIEW

src = np.float32([[830,270], [82,701], [1872,279], [2616, 701]])
dst = np.float32([[0, 0], [0, IMAGE_H], [IMAGE_W, 0], [IMAGE_W, IMAGE_H]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

img = cv2.imread('../test_img.png') 
print(img.shape)

warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
plt.imshow(warped_img)
plt.show()

#### DRAWING ON IMAGES

img = cv2.rectangle(img, (384,0),(510,128),(0,255,0), 3)
plt.imshow(img)
plt.show()


