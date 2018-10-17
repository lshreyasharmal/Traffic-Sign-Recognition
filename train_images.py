import cv2
import os
import numpy as np 
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_images_from_folder(folder):
    images = []
    c = 0
    for filename in os.listdir(folder):
    	c+=1
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
        	# images.append()
            images.append(rgb2gray(img))
        if(c==770):
        	break
    return images

path = "/home/mypc/Desktop/7th Sem/IA/Project V2/trainingset"
all_images = load_images_from_folder(path)
print all_images.shape