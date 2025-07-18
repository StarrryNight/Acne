import pickle
import numpy as np
import torch
from torch import nn 
import pandas as pd
classifier_types = ['blackheads', 'whiteheads', 'nodules', 'dark spot', 'pustules']
X = []
y = []

for idx, name in enumerate(classifier_types):
    with open(f'result/{name}_positives.pickle', 'rb') as f:
        positives = pd.DataFrame(pickle.load(f), dtype="object")
        print(positives.head)
    # Each positive is (img, info)
        
    for img, _ in positives:
        X.append(img)
        print(img)
        y.append(name)  # Assign class index

# Optionally, add negatives as a "background" class
# background_label = len(classifier_types)
# for name in classifier_types:
#     with open(f'result/{name}_negatives.pickle', 'rb') as f:
#         negatives = pickle.load(f)
#     for img, _ in negatives:
#         X.append(img)
#         y.append(background_label)

X = np.array(X)
y = np.array(y)

print(y)