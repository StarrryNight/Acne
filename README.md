# Acne Detector Region Proposal Pipeline

## Introduction

This project is designed to process and prepare image data for training convolutional neural networks (CNNs) to detect various types of acne lesions (such as blackheads, whiteheads, nodules, dark spots, and pustules) in facial images. The pipeline uses region proposal techniques and annotation data to extract and save candidate regions for each acne type, making it easier to train robust object detection models.

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
pip install numpy pandas matplotlib imageio scikit-image keras tensorflow
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
python processing.py
```

The script is robust to missing files and will print helpful error messages if it encounters issues.

## Output

After running, you will find the following files in the `result/` directory for each acne type:
- `{type}_positives.pickle`: Contains a list of warped image regions (NumPy arrays) for that acne type
- `{type}_infos.pickle`: Contains a list of metadata (image index and bounding box info) for each region

These files can be loaded in your training pipeline to feed directly into a CNN.

