import numpy as np
import cv2
from PIL import Image, ImageEnhance
import sys
import scipy
from scipy import ndimage
import math
from matplotlib import pyplot as plt
title = r"\speed_limit_40_rain"
directory = r"C:\Users\MyPC\Desktop\Images\CroppedImages"
path = directory+title
import os
if not os.path.exists(path):
    os.makedirs(path)
if not os.path.exists(path+"\Circle"):
    os.makedirs(path+"\Circle")
if not os.path.exists(path+"\Triangle"):
    os.makedirs(path+"\Triangle")
if not os.path.exists(path+"\Octagon"):
    os.makedirs(path+"\Octagon")
img = cv2.imread(r'C:\Users\MyPC\Desktop\Images'+title+'.jpg')


cv2.imshow('img',img)
#print("Adding Motion Blur")
#size = 15

# generating the kernel
##kernel_motion_blur = np.zeros((size, size))
##kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
##kernel_motion_blur = kernel_motion_blur / size
##
##print(kernel_motion_blur.shape)
##print(kernel_motion_blur)


# # applying the kernel to the input image

kernel2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel3 = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]])
kernel3 = kernel3/(-256)
output = cv2.filter2D(img, -1, kernel3)
cv2.imshow('enhanced',output)
img = output

# print("Adaptive Histogram Equalization")
bgr=img
#convert image from rgb to lab - l:lightness, a:green-red and b-blue yellow, numerical values
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#split into l,a,b
lab_planes = cv2.split(lab)
#clahe on lightness
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
#merge with image
lab = cv2.merge(lab_planes)
#convert back to bgr
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
img=bgr

img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)


#covert to graysale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#do thresholding
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
image, contours, h = cv2.findContours(thresh,1,2)
number=1
for i, cnt in enumerate (contours):
 approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

 if len(approx) > 10:
     #circle sign
     x, y, w, h = cv2.boundingRect(cnt)
     wdth, ht, g = img.shape
    
     x1 = int(x-0.1*w)
     y1 = int(y-0.1*h)
     h1 = int(h+h*0.2)
     w1 = int(w+0.2*w)
     if(x1>=0):
         x = x1
     if(y1>=0):
         y = y1
     if(w<=wdth):
         w = w1
     if(h<=ht):
         h = h1
    
     try:
         crop_img = img[y:y+h, x:x+w]
         img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
         mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
         mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
         mask = cv2.bitwise_or(mask1, mask2 )
         output = cv2.bitwise_and(crop_img,crop_img,mask=mask)
         if(not np.all(output==0) and np.count_nonzero(output)>0.1*output.shape[0]*output.shape[1]):
             cv2.imwrite(path+r"\Circle\Crop_img_"+str(number)+".png",crop_img)
             number+=1
        
##         mask = cv2.inRange(img_hsv, (90,0,0), (140,255,255))
##         output = cv2.bitwise_and(crop_img,crop_img,mask=mask)
##
##         if(not np.all(output==0) and np.count_nonzero(output)>0.1*output.shape[0]*output.shape[1]):
##             cv2.imwrite(path+r"\Circle\Crop_img_"+str(number)+".png",crop_img)
##             number+=1
     except:
         print("error")
 if len(approx)== 3:
     #triangle sign
     x, y, w, h = cv2.boundingRect(cnt)
     wdth, ht, g = img.shape
    
     x1 = int(x-0.1*w)
     y1 = int(y-0.1*h)
     h1 = int(h+h*0.2)
     w1 = int(w+0.2*w)
     if(x1>=0):
         x = x1
     if(y1>=0):
         y = y1
     if(w<=wdth):
         w = w1
     if(h<=ht):
         h = h1
        
     try:
         crop_img = img[y:y+h, x:x+w]
         img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
         mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
         mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
         mask = cv2.bitwise_or(mask1, mask2 )
         output = cv2.bitwise_and(crop_img,crop_img,mask=mask)

         if(not np.all(output==0) and np.count_nonzero(output)>0.1*output.shape[0]*output.shape[1]):
             cv2.imwrite(path+r"\Triangle\Crop_img_"+str(number)+".png",crop_img)
             number+=1
       
        
##         mask = cv2.inRange(img_hsv, (90,0,0), (140,255,255))
##         output = cv2.bitwise_and(crop_img,crop_img,mask=mask)
##
##         if(not np.all(output==0) and np.count_nonzero(output)>0.5*output.shape[0]*output.shape[1]):
##             cv2.imwrite(path+r"\Triangle\Crop_img_"+str(number)+".png",crop_img)
##             number+=1
     except:
         print("error")
 if len(approx)== 8:
     #triangle sign
     x, y, w, h = cv2.boundingRect(cnt)
     wdth, ht, g = img.shape
    
     x1 = int(x-0.1*w)
     y1 = int(y-0.1*h)
     h1 = int(h+h*0.2)
     w1 = int(w+0.2*w)
     if(x1>=0):
         x = x1
     if(y1>=0):
         y = y1
     if(w<=wdth):
         w = w1
     if(h<=ht):
         h = h1
        
     try:
         crop_img = img[y:y+h, x:x+w]
         img_hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
         mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
         mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
         mask = cv2.bitwise_or(mask1, mask2 )
         output = cv2.bitwise_and(crop_img,crop_img,mask=mask)

         if(not np.all(output==0) and np.count_nonzero(output)>0.1*output.shape[0]*output.shape[1]):
             cv2.imwrite(path+r"\Octagon\Crop_img_"+str(number)+".png",crop_img)
             number+=1
     except:
         print("error")

