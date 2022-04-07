import os
import matplotlib.pyplot as plt
import cv2 as cv

imagepath = "IMG20211126135232.jpeg"

img = cv.imread(imagepath)
rawfile = open(imagepath)
h,w,c = img.shape


#print(img[0,:,0])
plt.plot(img[:,:,0])


plt.show()

print(h,w,c)
#cv.imshow('test',img)
