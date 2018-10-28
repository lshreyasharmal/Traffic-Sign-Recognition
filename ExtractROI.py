import numpy as np
import cv2
from PIL import Image, ImageEnhance
import sys
import math
from matplotlib import pyplot as plt
title = "\mini_roundabout"
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


#color segmentation 
#img_hsi=img
# img_hsi = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
# # cv2.imshow('img_hsi',imag_hsi)

#mask1 = cv2.inRange(img_hsi, (17, 15, 100), (50, 56, 200))
# mask2 = cv2.inRange(img_hsi, (86, 31, 4), (220, 88, 50))
# mask3 = cv2.inRange(img_hsi, (25, 146, 190), (62, 174, 250))
# mask4 = cv2.inRange(img_hsi, (103, 86, 65), (145, 133, 128))
#print(mask)
#mask = mask1
# +mask2+mask3+mask4

#output = cv2.bitwise_and(img_hsi,img_hsi,mask=mask)
# output = img_hsi.copy()
# output[np.where(mask==0)] = 0
#cv2.imshow("img",output)


#define the list of boundaries
# boundaries = [
# 	([17, 15, 100], [50, 56, 200]), #red
# 	([86, 31, 4], [220, 88, 50]),
# 	([25, 146, 190], [62, 174, 250]),
# 	([103, 86, 65], [145, 133, 128])
# ]


# # loop over the boundaries
# for (lower, upper) in boundaries:
#        # create NumPy arrays from the boundaries
#        lower = np.array(lower, dtype = "uint8")
#        upper = np.array(upper, dtype = "uint8")

#        # find the colors within the specified boundaries and apply
#        # the mask
#        mask = cv2.inRange(img, lower, upper)
#        output = cv2.bitwise_and(img, img, mask = mask)

#        # show the images
#        cv2.imshow("images", np.hstack([img, output]))
#        cv2.waitKey(0)




print("sarthika")
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
            
            mask = cv2.inRange(img_hsv, (90,0,0), (140,255,255))
            output = cv2.bitwise_and(crop_img,crop_img,mask=mask)

            if(not np.all(output==0) and np.count_nonzero(output)>0.1*output.shape[0]*output.shape[1]):
                cv2.imwrite(path+r"\Circle\Crop_img_"+str(number)+".png",crop_img)
                number+=1
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
           
            
            mask = cv2.inRange(img_hsv, (90,0,0), (140,255,255))
            output = cv2.bitwise_and(crop_img,crop_img,mask=mask)

            if(not np.all(output==0) and np.count_nonzero(output)>0.5*output.shape[0]*output.shape[1]):
                cv2.imwrite(path+r"\Triangle\Crop_img_"+str(number)+".png",crop_img)
                number+=1
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
  
