# Acne Detector Region Proposal Pipeline
**Latest model accuracy -- 50%

loss and accuracy graphs:
<img width="1910" height="558" alt="image" src="https://github.com/user-attachments/assets/700ef095-7fb9-45eb-bd54-2ad9a00907af" />

## Introduction (Image is taken from roboflow universe and processed through this model)
<img width="1067" height="406" alt="image" src="https://github.com/user-attachments/assets/70282dc5-3c31-4151-ba17-875673f15290" />

This project is designed to process and prepare image data for training convolutional neural networks (CNNs) to detect various types of acne lesions (such as blackheads, whiteheads, nodules, dark spots, and pustules) in facial images. The pipeline uses region proposal techniques and annotation data to extract and save candidate regions for each acne type, making it easier to train robust object detection models.

Instead of a traditional CNN, the project uses RCNN, regional convolutional networks, to train and identify acne on pictures provided. The process begins by creating regional proposals using Felzenszwalb's graph based segmentation algorithm. These regions are later grouped through intersection union find through similarities between regions. Then these regions are processed into tensors, which are used to train the CNN.

This project is based on Rich feature hierarchies for accurate object detection and semantic segmentation by Ross Girshick et al, and is modified upon ObjectDetectionWithRCNN from Yumi Kondo on github. 

Model is trained using dataset from https://universe.roboflow.com/acne-training-9ct8c/acne-training-3
## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- imageio
- scikit-image
- keras
- tensorflow

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Data Preparation

1. **Images:** Place your preprocessed training images in the `preprocessed_data/train/` directory. Each image should be named according to its `file_ID` (as referenced in the annotation CSV) with a `.jpg` extension.

2. **Annotations:** Ensure you have an annotation CSV file named `annotation_df.csv` in the `preprocessed_data/` directory. This file should contain bounding box information and object types for each image.

## Running the Processing Script

The main script for processing is `processing.py`. It will:
- Load the annotation data and images
- Use region proposal methods to generate candidate regions
- Match regions to annotated objects using IoU
- Warp and save only one region per object (per image) for each acne type
- Save the resulting image regions and their metadata to the `result/` directory

To run the script:

```bash
python main.py
```

Model is already trained and saved in the result folder as classifier. Changing the image path in main.py allow different images to be tested.

## Output

After running, you will find the following files in the `result/` directory for each acne type:
- `{type}_positives.pickle`: Contains a list of warped image regions (NumPy arrays) for that acne type
- `{type}_infos.pickle`: Contains a list of metadata (image index and bounding box info) for each region

These files can be loaded in your training pipeline to feed directly into a CNN.

