import numpy as np
import pandas as pd
import os, sys 
import matplotlib.pyplot as plt
import random
import imageio
import skimage
from keras.applications import VGG16
from keras import models
import rg_proposal as rg
import pickle
import helper as h
import time 

modelvgg16 = VGG16(include_top=True,weights='imagenet')
modelvgg = models.Model(inputs  =  modelvgg16.inputs, 
                        outputs = modelvgg16.layers[-3].output)
img_8bit = imageio.imread("preprocessed_data/train/01F3MMV6NBB013AEV2SH04SRQ9_jpeg.rf.a88bc38371062a738c5591586ce56307.jpg")
regions = rg.selective_search(img_8bit)
def warp(img, newsize):
    img_resize = skimage.transform.resize(img,newsize)
    return(img_resize)
img_dir = "preprocessed_data/train"


warped_size = (224, 224)
X = []
for i,x in enumerate([1511,1654,1713,1692,1757]):
    r = regions[i]
    ## extract a single candidate region
    x , y , width, height = r["rect"]
    img_bb     = img_8bit[y:y + height,x:x + width]
    ## warp image
    img_warped = warp(img_bb, warped_size)
    ## create CNN feature
    feature = modelvgg.predict(img_warped.reshape(1,warped_size[0],warped_size[1],3))
    
    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(1,3,1)
    #ax.imshow(img_bb)
    ax.set_title("region proposal, shape={}".format(img_bb.shape))
    
    ax = fig.add_subplot(1,3,2)
    #ax.imshow(img_warped)
    ax.set_title("warped image, shape={}".format(img_warped.shape))

    ax = fig.add_subplot(1,3,3)    
    #ax.imshow(feature, cmap='hot')
    ax.set_title("feature length = {}".format(len(feature.flatten())))
    #plt.show()



annotation_df = pd.read_csv("preprocessed_data/annotation_df.csv")


IoU_cutoff_object     = 0.6
IoU_cutoff_not_object = 0.4
objnms = ["image0","info0","image1","info1","image1","info1","image1","info1","image1","info1","image1","info1"]  
dir_result = "result"



start = time.time()   
print("o")
# the "rough" ratio between the region candidate with and without objects.
N_img_without_obj = 2 
newsize = (300,400) ## hack
# --- REPLACE THE OLD image0, info0, image1, info1 with a dictionary for each classifier type ---
classifier_types = ['blackheads', 'whiteheads', 'nodules', 'dark spot', 'pustules']
region_data = {name: {'positives': [], 'infos': []} for name in classifier_types}
print(annotation_df.shape[0])
for irow in range(annotation_df.shape[0]):
    try:
        # extract a single frame that contains at least one object of interest (classifier)
        row  = annotation_df.iloc[irow,:]
        # read in the corresponding frame
        path = os.path.join(img_dir,row["file_ID"] + ".jpg")
        print(irow)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        img  = imageio.imread(path)
        if img.ndim != 3 or img.shape[2] != 3:
            print(f"Image at {path} does not have 3 channels, shape: {img.shape}")
            continue
        orig_h, orig_w, _ = img.shape
        # resize all the images into newsize = (200,250)
        img  = warp(img, newsize)
        orig_nh, orig_nw, _ = img.shape
        # region candidates for this frame
        regions = h.get_region_proposal(img,min_size=50)[::-1]
        # for each object that exists in the data,
        for ibb in range(row["object_num"]): 
            name = row["bbx_{}_name".format(ibb)]
            # Only process if name is in classifier_types
            if name not in classifier_types:
                continue
            #if irow % 50 == 0:
            print(f"frameID = {irow:04.0f}/{annotation_df.shape[0]}, BBXID = {ibb:02.0f},  N region proposals = {len(regions)}, N positives gathered till now = {len(region_data.get(name, {}).get('positives', []))}")
            # extract the bounding box of the classifier object  
            multx, multy  = orig_nw/orig_w, orig_nh/orig_h 
            true_xmin     = row["bbx_{}_xmin".format(ibb)]*multx
            true_ymin     = row["bbx_{}_ymin".format(ibb)]*multy
            true_xmax     = row["bbx_{}_xmax".format(ibb)]*multx
            true_ymax     = row["bbx_{}_ymax".format(ibb)]*multy
            _positive = None
            _positive_info = None
            print(_positive)
            # for each candidate region, find if this classifier object is included
            for r in regions:
                prpl_xmin, prpl_ymin, prpl_width, prpl_height = r["rect"]
                # check bounds
                if (prpl_xmin < 0 or prpl_ymin < 0 or
                    prpl_xmin + prpl_width > img.shape[1] or
                    prpl_ymin + prpl_height > img.shape[0]):
                    continue
                # calculate IoU between the candidate region and the classifier object
                IoU = h.get_IOU(prpl_xmin, prpl_ymin, prpl_xmin + prpl_width, prpl_ymin + prpl_height,
                                 true_xmin, true_ymin, true_xmax, true_ymax)
                if IoU ==0:
                    continue
                # candidate region numpy array
                img_bb = np.array(img[prpl_ymin:prpl_ymin + prpl_height,
                                      prpl_xmin:prpl_xmin + prpl_width])
                if img_bb.size == 0 or img_bb.shape[0] == 0 or img_bb.shape[1] == 0:
                    print(f"Empty region at row {irow}, bb: {prpl_xmin},{prpl_ymin},{prpl_width},{prpl_height}")
                    continue
                # Warp the candidate region to fixed size for CNN compatibility
                img_bb_warped = warp(img_bb, warped_size)
                info = [irow, prpl_xmin, prpl_ymin, prpl_width, prpl_height]
                if IoU > IoU_cutoff_object:
                    print('jo')
                    _positive = img_bb_warped
                    _positive_info = info
                    print(_positive)
                    # Only add the first valid region for this object
                    region_data[name]['positives'].append(_positive)
                    region_data[name]['infos'].append(_positive_info)
                    break
            # Do not add more than one region per object
    except Exception as e:
        print(f"Error at row {irow}: {e}")
        continue

        
end = time.time()  
print("TIME TOOK : {}MIN".format((end-start)/60))

### Save image0, info0, image1, info1 
# --- Save each classifier type's regions to separate pickle files ---
for name, data in region_data.items():
    with open(os.path.join(dir_result, f"{name}_positives.pickle"), 'wb') as handle:
        pickle.dump(data['positives'], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(dir_result, f"{name}_infos.pickle"), 'wb') as handle:
        pickle.dump(data['infos'], handle, protocol=pickle.HIGHEST_PROTOCOL)

# --- Optionally, update the plotting function to accept a list of (image, info) tuples ---
def plot_cadidate_regions_in_training(region_list, title):
    fig = plt.figure(figsize=(12,12))
    fig.subplots_adjust(hspace=0.0001,
                        wspace=0.0001,
                        left=0,right=1,bottom=0, top=1)
    print(title)
    nw, nh = 10, 10
    count = 1
    for irow in range(min(100, len(region_list))):
        im, info = region_list[irow]
        ax  = fig.add_subplot(nh,nw,count)
        ax.imshow(im)
        ax.axis("off")
        count += 1
    plt.show()

# Example usage for plotting (for a specific type, e.g., 'dark spot')
# plot_cadidate_regions_in_training(region_data['dark spot']['positives'], title="plot warped candidate regions with a dark spot object in training")
# No negatives to plot