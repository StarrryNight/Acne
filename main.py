import matplotlib.pyplot as plt
import imageio, os 
import skimage.transform
import numpy as np
import training.helper as rg
import seaborn as sns
from keras.applications import VGG16
from keras import models
import time
from keras.models import load_model


dir_image = "preprocessed_data/valid"
img = imageio.imread(os.path.join(dir_image,"01F3MMVCYP2RSAQXDX608JXCXX_jpeg.rf.f674829b29e4c8ff42be22b4bc8c7cfc.jpg"))
## resize the image because the original image is a bit too large and takes lots of time for computation
# I used this resizing hack to train the classifier and also to extract candidate regions
newsize = (200,250)
img = skimage.transform.resize(img,newsize)
const = 4
plt.figure(figsize=(5*const,6*const))
plt.imshow(img)
plt.show()

regions = rg.get_region_proposal(img,min_size=50)
print("N candidate regions ={}".format(len(regions)))
print("_"*10)
print("print the first 10 regions")
for r in regions[:10]:
    print(r)
print("_"*10)
print("print the last 10 regions")    
for r in regions[-10:]:
    print(r)

    
def plt_rectangle(plt,label,x1,y1,x2,y2,color = "yellow", alpha=0.5):
    linewidth = 3
    if type(label) == list:
        linewidth = len(label)*3 + 2
        label = ""
        
    plt.text(x1,y1,label,fontsize=20,backgroundcolor=color,alpha=alpha)
    plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color, alpha=alpha)
    
    
plt.figure(figsize=(20,20))    
plt.imshow(img)
for item, color in zip(regions,sns.xkcd_rgb.values()):
    x1, y1, width, height = item["rect"]
    label = item["labels"][:5]
    plt_rectangle(plt,label,
                  x1,
                  y1,
                  x2 = x1 + width,
                  y2 = y1 + height, 
                  color= color)
plt.show()


def warp_candidate_regions(img,regions):
    ## for each candidate region, 
    ## warp the image and extract features 
    newsize_cnn = (224, 224)
    X = []
    for i, r in enumerate(regions):
        origx , origy , width, height = r["rect"]
        candidate_region = img[origy:origy + height,
                               origx:origx + width]
        img_resize = skimage.transform.resize(candidate_region,newsize_cnn)
        X.append(img_resize)

    X = np.array(X)
    print(X.shape)
    return(X)
X = warp_candidate_regions(img,regions)

modelvgg16 = VGG16(include_top=True,weights='imagenet')
modelvgg16.summary()

modelvgg = models.Model(inputs  = modelvgg16.inputs, 
                        outputs = modelvgg16.layers[-3].output)
## show the deep learning model
modelvgg.summary()
start   = time.time()
feature = modelvgg.predict(X)
end     = time.time()
print("TIME TOOK: {:5.4f}MIN".format((end-start)/60.0))

dir_result = "result"
classifier = load_model(os.path.join(dir_result,"classifier.h5"))
classifier.summary()
y_pred = classifier.predict(feature)


classifier_types = ['blackheads', 'whiteheads', 'nodules', 'dark spot', 'pustules']
print(classifier_types[y_pred.index(max(y_pred))])