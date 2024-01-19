import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

img = cv2.imread('cube2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0.1)
edges = cv2.Canny(gray, 50, 180)

plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap = 'gray')
plt.title('Edges'), plt.xticks([]), plt.yticks([])

plt.show()