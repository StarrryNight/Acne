import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from collections import Counter

annotation_df = pd.read_csv("preprocessed_data/annotation_df.csv")
maxNobj = np.max(annotation_df["object_num"])
train_data_dir = "preprocessed_data/train"

#Show histogram of objects
def histogram():
    plt.hist(annotation_df["object_num"].values, bins = 100)
    plt.title("max N of objects per image={}".format(maxNobj))
    plt.show()


#Show diagram of number of objects
def count_diagram():    
    class_obj = []
    for ibbx in range(maxNobj):
        class_obj.extend(annotation_df["bbx_{}_name".format(ibbx)].values)
    class_obj = np.array(class_obj)

    count             = Counter(class_obj[class_obj != 'nan'])
    print(count)
    class_nm          = list(count.keys())
    class_count       = list(count.values())
    asort_class_count = np.argsort(class_count)

    class_nm          = np.array(class_nm)[asort_class_count]
    class_count       = np.array(class_count)[asort_class_count]

    xs = range(len(class_count))
    plt.barh(xs,class_count)
    plt.yticks(xs,class_nm)
    plt.title("The number of objects per class: {} objects in total".format(len(count)))
    plt.show()

#helper function for plotting
def plt_rectangle(plt,label,x1,y1,x2,y2):
    '''
    == Input ==
    
    plt   : matplotlib.pyplot object
    label : string containing the object class name
    x1    : top left corner x coordinate
    y1    : top left corner y coordinate
    x2    : bottom right corner x coordinate
    y2    : bottom right corner y coordinate
    '''
    linewidth = 3
    color = "yellow"
    plt.text(x1,y1,label,fontsize=20,backgroundcolor="magenta")
    plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color)
    plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color)
    plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color)
    plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color)


#shows images with plot
def plotOnImage():
    size = 20    
    ind_random = np.random.randint(0,annotation_df.shape[0],size=size)
    for irow in ind_random:
        row  = annotation_df.iloc[irow,:]
        path = os.path.join(train_data_dir, row["file_ID"] + ".jpg")
        # read in image
        img  = imageio.imread(path)

        plt.figure(figsize=(12,12))
        plt.imshow(img) # plot image
        plt.title("object_num={}, height={}, width={}".format(row["object_num"],row["height"],row["width"]))
        # for each object in the image, plot the bounding box
        for iplot in range(row["object_num"]):
            plt_rectangle(plt,
                        label = row["bbx_{}_name".format(iplot)],
                        x1=row["bbx_{}_xmin".format(iplot)],
                        y1=row["bbx_{}_ymin".format(iplot)],
                        x2=row["bbx_{}_xmax".format(iplot)],
                        y2=row["bbx_{}_ymax".format(iplot)])
        plt.show() ## show the plot



#Comment out to activate
#histogram()
#count_diagram()
#plotOnImage()