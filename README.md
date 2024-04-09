<div align="left">
  <img width="15%" src="/figures/pyrcellmech_logo_v2.png" alt="pycellmech Logo">
</div>

# PyCellMech

## Introduction

PyCellMech is designed to understand how shapes impact our understanding of medical conditions and/or their progression.

This package currently begins by introducing some preliminary shape-based analyses, with focus on geometric, polygonal and one-dimensional shape features.


## Requirements

Packages used in PyCellMech include:

```
    matplotlib
    numpy
    math
    opencv-python
    pandas
    scikit-learn

```

## Usage

PyCellMech takes binarized masks as its input. There are no restrictions in terms of 
regions of interest (ROI), as the method will iterate through each ROI
within each binarized image. Whether you have a single image or multiple images to process,
binarized masks should be contained within a folder. 

Once you have your input images ready, download the code, head to the ``main.py`` 
file, and change the following settings (located on lines 176-178)::

```
    folder_path = '/path/to/binarized/images'
    csv_file_path = '/path/to/save/shape_features.csv'
    output_folder = '/path/to/save/feature/maps'
```

``csv_file_path`` will reflect the location you would like to save the CSV file, and ``output_folder`` is where the feature visualization map for every image processed will be saved. 

Feature visualization maps select the largest ROI and superimpose the extracted features on the ROI. Thus, this package provides quantitative and qualitative assessments. 

