import os 
import numpy as np
import xml.etree.ElementTree as ET
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

def extract_single_xml_file(tree):
    object_num = 0
    row = OrderedDict()
    for elems in tree.iter():
        if elems.tag == "size":
            for elem in elems:
                row[elem.tag] = int(elem.text)
        if elems.tag == "object":
            for elem in elems:
                if elem.tag == "name":
                    row["bbx_{}_{}".format(object_num,elem.tag)] = str(elem.text)              
                if elem.tag == "bndbox":
                    for k in elem:
                        row["bbx_{}_{}".format(object_num,k.tag)] = float(k.text)
                    object_num += 1
    row["object_num"] = object_num
    return(row)

annotation_df = []

for fnm in os.listdir("data/train"):
    if fnm.endswith('.xml'):
        tree = ET.parse(os.path.join("data/train", fnm))
        row = extract_single_xml_file(tree)
        row["file_ID"] = fnm.split(".")[0]
        annotation_df.append(row)

annotation_df = pd.DataFrame(annotation_df)

max_object_num =np.max(annotation_df["object_num"])
print("columns in df_anno\n-----------------")
for icol, colnm in enumerate(annotation_df.columns):
    print("{:3.0f}: {}".format(icol,colnm))
print("-"*30)
print("df_anno.shape={}=(N frames, N columns)".format(annotation_df.shape))
print(annotation_df.head())