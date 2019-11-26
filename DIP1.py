import cv2
import time
print(cv2.__version__)

img = cv2.imread('Image.jpeg', 0)
cv2.imshow('img', img)
cv2.waitKey(1)
time.sleep(3)
cv2.circle(img, (50, 50), 40, (140, 140, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
