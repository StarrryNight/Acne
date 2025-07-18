import os 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.misc
import skimage.segmentation
import skimage.feature
import imageio
from copy import copy
import training.helper as h
img_dir = "preprocessed_data/train"

############MAIN STARTS HERE#######################
#segtmentize
scale    = 1.0
sigma    = 0.8
min_size = 500

np.random.seed(5)
listed_path = os.listdir(img_dir)
Nplot = 5
random_img_path = np.random.choice(listed_path,Nplot)
for imgnm in random_img_path:
    if imgnm[-1]=='g':
    # import 8 bits degital image (each digit ranges between 0 - 255)
        img_8bit  = imageio.imread(os.path.join(img_dir,imgnm))
        img       = h.image_segmentation(img_8bit, scale, sigma, min_size)
    
        fig = plt.figure(figsize=(15,30))
        ax  = fig.add_subplot(1,2,1)
        #ax.imshow(img_8bit)
        ax.set_title("original image")
        ax  = fig.add_subplot(1,2,2)
        #ax.imshow(img[:,:,3])
        ax.set_title("skimage.segmentation.felzenszwalb, N unique region = {}".format(len(np.unique(img[:,:,3]))))
        #plt.show()




R = h.extract_region(img)
print("{} rectangle regions are found".format(len(R)))

#GTenerate regionmaps
figsize = (20,20)
plt.figure(figsize=figsize)    
#plt.imshow(img[:,:,:3]/2**8)
for item, color in zip(R.values(),sns.xkcd_rgb.values()):
    x1 = item["min_x"]
    y1 = item["min_y"]
    x2 = item["max_x"]
    y2 = item["max_y"]
    label = item["labels"][0]
    h.plt_rectangle(plt,label,x1,y1,x2,y2,color=color)
#plt.show()

plt.figure(figsize=figsize)    
#plt.imshow(img[:,:,3])
for item, color in zip(R.values(),sns.xkcd_rgb.values()):
    x1 = item["min_x"]
    y1 = item["min_y"]
    x2 = item["max_x"]
    y2 = item["max_y"]
    label = item["labels"][0]
    h.plt_rectangle(plt,label,x1,y1,x2,y2,color=color)
#plt.show()



#generate texture map
tex_grad =h.calc_texture_gradient(img)   
h.plot_image_with_min_max(tex_grad,nm="tex_grad")


#Generate hsv map
hsv = h.calc_hsv(img)
h.plot_image_with_min_max(hsv,nm="hsv")


R = h.augment_regions_with_histogram_info(tex_grad, img,R,hsv,tex_grad)

neighbours = h.extract_neighbours(R)
print("Out of {} regions, we found {} intersecting pairs".format(len(R),len(neighbours)))

print("S[(Pair of the intersecting regions)] = Similarity index")
S = h.calculate_similarlity(img,neighbours,verbose=True)

regions = h.merge_regions_in_order(S,R,img.shape[0]*img.shape[1],verbose=True)


plt.figure(figsize=(20,20))    
#plt.imshow(img[:,:,:3]/2**8)
for item, color in zip(regions,sns.xkcd_rgb.values()):
    x1, y1, width, height = item["rect"]
    label = item["labels"][:5]
    h.plt_rectangle(plt,label,x1,y1,x2 = x1 + width,y2 = y1 + height, color = color)
#plt.show()

plt.figure(figsize=(20,20))    
#plt.imshow(img[:,:,3])
for item, color in zip(regions,sns.xkcd_rgb.values()):
    x1, y1, width, height = item["rect"]
    label = item["labels"][:5]
    h.plt_rectangle(plt,label,
                  x1,
                  y1,
                  x2 = x1 + width,
                  y2 = y1 + height, color= color)
#plt.show()


def selective_search(im): 
    regions = h.get_region_proposal(im,min_size=500)
    return regions