<div align="left">
  <img width="15%" src="/figures/pyrcellmech_logo_v2.png" alt="pycellmech Logo">
</div>

# PyCellMech

## Introduction

PyCellMech is designed to understand how shapes impact our understanding of medical conditions and/or their progression.

This package currently begins by introducing some preliminary shape-based analyses, with focus on geometric, polygonal and one-dimensional shape features.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. See the [LICENSE](./LICENSE) file for details.

## Installation
```
pip install pycellmech
```

## Usage

PyCellMech takes binarized masks as its input. There are no restrictions in terms of 
regions of interest (ROI), as the method will iterate through each ROI
within each binarized image. Whether you have a single image or multiple images to process,
binarized masks should be contained within a folder. 

Once you have your input images ready, use the following command line:

```
pycellmech --input /path/to/input --csv_file /path/to/saving/csv --output /path/to/saving/output/images --label 's' for single-class and 'm' for multi-class --nifti_folder /path/to/nifti/files
```

``csv_file`` will reflect the location you would like to save the CSV file, and ``output`` is where the feature visualization map for every image processed will be saved, ``label``is where you specify whether your features are extracted with single or multiple labels, and if the multi-class option is selected, then the end user need to specify where NifTI files are contained using ``nifti_folder``.

To create NifTI-based metadata for labeling multi-class binary files, the following command lines can be used:

````
    pycellmech_create_label 
    --folder_path /path/to/folder/with/masks
    --output_csv_folder /path/to/save/csv/label 
    --output_image_folder /path/to/save/labeled/masks
````

````
    pycellmech_nifti 
    --folder_path /path/to/folder/with/masks
    --input_csv_folder /path/to/extract/csv/labels
    --nifti_save_dir /path/to/save/nifti/file
    --label_save_dir /path/to/save/updated/labeled/images
````

Feature visualization maps select the largest ROI and superimpose the extracted features on the ROI. Thus, this package provides quantitative and qualitative assessments. 

A handful of samples from the Kvasir dataset have been included in this repository for testing. 


## Shape Features

The current iteration of PyCellMech involves three shape-based feature classes: 

    - One-dimensional function shape features
    - Geometric Shape Features
    - Polygonal Approximation Shape Features

These feature classes will continue to be built upon in future iterations. A summary of 
these feature descriptions are found below. 

### One-dimensional function shape features

*Complex Coordinate*

A complex coordinate function derives a complex number from a point on an object's contour.

*Centroid Distance Function (CDF)*

CDF the distance from the centroid of an object to points on its boundary, providing a way to describe the shape's geometry relative to its center.

*Area Function (AF)*

AF quantifies the total surface area enclosed by the boundaries of the shape.

*Triange Area Representation (TAR)*

TAR calculates the area of triangles formed within or around the shape.

*Chord Length Function (CLF)*

CLF measures the distances between pairs of points on the shape's boundary.

### Geometric Shape Features

*Axis Minimum Inertia (AMI)*

AMI identifies the axis along which the shape exhibits the least resistance to rotational motion.

*Average Bending Energy (ABE)*

ABE quantifies the mean amount of effort required to deform the shape.

*Eccentricity*

Eccentricity describes the degree to which a shape deviates from being circular.

*Minimum Bounding Rectangle (MBR)*

MBR represents the smallest rectangle that entirely encloses the shape, representing its rectilinear approximation. 

*Circularity Ratio (CR)*

CR compares the area of the shape to the area of a circle with the same perimeter.

*Ellipse Variance and Moment Invariants (EM and EV)*

EV measures the deviation of a shape from an elliptical form, and EM quantifies shape characteristics that remain constant under transformations (e.g., rotation, scaling, and translation).

*Solidity*

Solidity calculates the ratio of the shape's area to the area of its convex hull.

### Polygonal Approximation Shape Features

*Distance Threshold Method (DTM)*

DTM  involves setting a specific distance limit to differentiate between relevant and irrelevant points.

*Polygon Evolution by Vertex Deletion (PEVD)*

PEVD is a process where vertices are incrementally removed from a polygonal shape to simplify its structure while trying to preserve its overall form and characteristics.

*Splitting Method (SM)*

SM involves dividing a shape into smaller, manageable segments or components.

*Minimum Perimeter Polygon (MPP)*

MPP is the polygon with the smallest possible perimeter that can enclose a given shape.

*K-Means Method*

KMeans clusters points on the shape into a specified number of groups based on their proximity.


## References

The PyCellMech package was inspired by the below works. The package will be continually updated based
on the latest research developments. 

```
P.-Y. Yin. Pattern Recognition Techniques, Technology and Applications. InTech, November
2008. ISBN 9789537619244. doi: 10.5772/90.
```

```
J. Chaki and N. Dey. A Beginners Guide to Image Shape Feature Extraction Techniques.
Taylor Francis Group, 2019. ISBN 9781000034301. doi: 10.5772/90.
```

Kvasir Dataset 
```
Konstantin Pogorelov, Kristin Ranheim Randel, Carsten Griwodz, Sigrun Losada Eskeland,
Thomas de Lange, Dag Johansen, Concetto Spampinato, Duc-Tien Dang-Nguyen, Math-
ias Lux, Peter Thelin Schmidt, Michael Riegler, and Pal Halvorsen. Kvasir: A multi-class
image dataset for computer aided gastrointestinal disease detection. In Proceedings of the
8th ACM on Multimedia Systems Conference, MMSys’17, pages 164–169, New York, NY,
USA, 2017. ACM. ISBN 978-1-4503-5002-0. doi: 10.1145/3083187.3083212.
```

## Citation
```
@inproceedings{arslan2024pycellmech,
      url = {https://spie.org/medical-imaging/presentation/PyCellMech--A-shape-based-feature-extraction-pipeline-for-use/13406-83#_=_},
      title={PyCellMech: A shape-based feature extraction pipeline for use in medical and biological studies}, 
      author={Janan Arslan and Henri Chhoa and Ines Khemir and Romain Valabregue and Kurt K. Benke},
      publisher = {SPIE},
      editor = {Mitra, Jhimli and Colliot, Olivier},
      year = {2025},
      month = feb
}
```
