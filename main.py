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
import training.helper as h

dir_image = "preprocessed_data/valid"
img = imageio.imread(os.path.join(dir_image,"levle3_57_jpg.rf.4677ffc10f5281e0f58e19e433afc7f6.jpg"))
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

    


X = h.warp_candidate_regions(img,regions)

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
def plot_selected_regions_with_estimated_prob(y_pred,
                                              method="highest",
                                              upto=5):
    ## increasing order
    irows = np.argsort(y_pred[:,0])
    if method == "highest":
        irows = irows
    count = 1
    const = 4
    fig = plt.figure(figsize=(5*const,int(np.ceil(upto/5))*const))
    fig.subplots_adjust(hspace=0.13,wspace=0.0001,
                        left=0,right=1,bottom=0, top=1)
    for irow in irows:
        print(y_pred[irow])
        prob = np.max(y_pred[irow])
        typee = classifier_types[np.argmax(y_pred[irow])]
        r    = regions[irow]
        origx , origy , width, height = r["rect"]
        
        ax = fig.add_subplot(int(np.ceil(upto/5)),5,count)
        ax.imshow(img)
        ax.axis("off")
        h.plt_rectangle(ax,label="",
                      x1=origx,
                      y1=origy,
                      x2=origx + width,
                      y2=origy+height,color = "yellow", alpha=0.5)
        
        #candidate_region = img[origy:origy + height,
        #                      origx:origx + width]       
        #ax.imshow(candidate_region)
        ax.set_title("Prob={:4.3f} type ={}".format(prob, typee))
        count += 1
        if count > upto:
            break
    plt.show()
print("The most likely candidate regions")    
plot_selected_regions_with_estimated_prob(y_pred,method="highest",upto=5)

