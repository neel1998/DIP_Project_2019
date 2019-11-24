import cv2
import numpy as np
import matplotlib.pyplot as plt

a = cv2.imread('fil.png')
print(a.shape)
plt.imshow(a)