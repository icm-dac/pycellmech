'''
======================================================================
 Title:                   PYCELLMECH
 Creating Author:         Janan Arslan
 Creation Date:           22 FEB 2024
 Latest Modification:     15 MAY 2024
 Modification Author:     Janan Arslan
 E-mail:                  janan.arslan@icm-institute.org
 Version:                 2.1.12
======================================================================


pycellmech is designed to extract appropriate shape features which can
be used to extrapolated how the shape of objects in medical studies can
be used to elucidate the mechanics of the disease and/or its progression.

Input for pycellmech is current binarized images. The images can contain one
or multiple regions of interest. Features will be extracted for all contours
within the image and saved as a CSV file. For the visualization of these features,
the largest contour from each image is selected and all features are visualized according
to this contour. The feature maps are saved for each image. 

'''

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import nibabel as nib
import tempfile
from skimage.measure import label, regionprops


from .one_dimensional_features import (
    get_one_dimensional_features, plot_original_contour, plot_centroid, 
    plot_complex_coordinates, plot_cdf, plot_area_function, plot_clf, plot_tar
)
from .geometric_shape_features import (
    get_geometric_shape_features, plot_AMI, plot_ABE, plot_eccentricity, 
    plot_mbr, plot_circularity_ratio, plot_ellipse_features, plot_solidity
)
from .polygonal_shape_features import (
    get_polyognal_shape_features, plot_DTM, plot_PEVD, plot_SM, plot_MPP, plot_KMeans
)


# Preprocess image and find contours
def preprocess_image(binary_image_path):
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(binary_image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary_image

# Preprocess image and find contours
def preprocess_slice(slice_data, label_value):
    binary_image = np.uint8(slice_data == label_value) * 255
    blur = cv2.GaussianBlur(binary_image, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary_image

# Process NiFfty files
def preprocess_nii_image(nii_image_path):
    try:
        nii_img = nib.load(nii_image_path)
        nii_data = nii_img.get_fdata()
        print(f'Loaded NIfTI file {nii_image_path} with shape {nii_data.shape}')
    except Exception as e:
        print(f'Error loading NIfTI file {nii_image_path}: {e}')
        return [], {}

    contours = []
    binary_images = {}

    if len(nii_data.shape) == 2:  # 2D case
        for label_value in np.unique(nii_data):
            if label_value == 0:  # Skip background
                continue
            label_contours, binary_image = preprocess_slice(nii_data, label_value)
            for contour in label_contours:
                contours.append((contour, label_value))
            binary_images[label_value] = binary_image
    else:
        print(f'NIfTI file {nii_image_path} does not have the expected 2D shape.')
        return [], {}

    return contours, binary_images

# Feature illustration through largest contour
def plot_features(binary_image, contour, one_d_features, geom_features, poly_features, filename, output_folder, label=None):
    fig, axs = plt.subplots(4, 7, figsize=(45, 8), constrained_layout=False)
    fig.set_constrained_layout_pads(w_pad=4, h_pad=4, hspace=0.2, wspace=0.2)

    fig.suptitle('Shape Feature Visualization', fontsize=16)

    # Calculate moments for contour
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    # Plot original contour as reference
    plot_original_contour(axs[0, 0], binary_image)

    # Plot centroid
    plot_centroid(axs[0, 1], binary_image, cx, cy)

    # Plot complex coordinates
    complex_coordinates = one_d_features['complex_coordinates']
    plot_complex_coordinates(axs[0, 2], binary_image, cx, cy, complex_coordinates)

    # Plot CDF
    cdf = one_d_features['cdf']
    plot_cdf(axs[0, 3], binary_image, cdf, cx, cy, contour)

    # Plot AF
    plot_area_function(axs[0, 4], binary_image, cx, cy, contour)
    tangent_vectors = one_d_features['tangent_vectors']

    # Plot CLF
    plot_clf(axs[0, 5], binary_image, contour, tangent_vectors)

    # Plot TAR
    tar_values = one_d_features['tar_values']
    plot_tar(axs[0, 6], binary_image, contour, tar_values)

    for row in axs:
        for ax in row:
            ax.axis('off')

    axs[1, 0].axis('on')

    # Plot AMI
    alpha = geom_features['alpha']
    plot_AMI(axs[1, 0], binary_image, contour, cx, cy, alpha)

    # Plot ABE
    curvature_list = one_d_features['curvature']
    abe_value = geom_features['abe']
    plot_ABE(axs[1, 1], binary_image, contour, abe_value, curvature_list)

    # Plot Eccentricity
    eccentricity = geom_features['eccentricity']
    plot_eccentricity(axs[1, 2], contour, cx, cy, eccentricity)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot MBR
    mbr_center, mbr_width, mbr_height, mbr_angle, E, Elo = geom_features['mbr_center'], geom_features['mbr_width'], geom_features['mbr_height'], geom_features['mbr_angle'], geom_features['mbr_E'], geom_features['elongation']
    contour_array = contour.squeeze()
    axs[1, 3].axis('on')
    axs[1, 3].invert_yaxis()
    plot_mbr(axs[1, 3], contour_array, cv2.minAreaRect(contour_array))

    # Plot CR
    circularity_ratio = geom_features['circularity_ratio']
    axs[1, 4].axis('on')
    axs[1, 4].invert_yaxis()
    plot_circularity_ratio(axs[1, 4], contour_array, circularity_ratio)

    # Plot EV and EM
    ev_value, em_value = geom_features['EV'], geom_features['EM']
    axs[1, 5].axis('on')
    axs[1, 5].invert_yaxis()
    plot_ellipse_features(axs[1, 5], contour_array, ev_value, em_value)

    # Plot solidity
    solidity, hull = geom_features['solidity'], geom_features['hull']
    for ax in axs.flat:
        ax.set_aspect('equal')
    plot_solidity(axs[1, 6], contour_array, solidity)


    # Plot DTM    
    # Flatten contour array
    contour_reshape = contour.reshape(-1, 2)
    dtm = poly_features['dtm']
    plot_DTM(axs[2, 0], contour_reshape, dtm)

    # Plot PEVD
    pevd = poly_features['pevd']
    plot_PEVD(axs[2, 1], contour_reshape, pevd)

    # Plot SM
    sm = poly_features['sm']
    plot_SM(axs[2, 2], contour_reshape, sm)

    # Plot MPP
    mpp = poly_features['mpp']
    plot_MPP(axs[2, 3], contour_reshape, mpp)

    # Plot KMeans
    km_cluster_centers, km_line_segments, km_labels = poly_features['km_cc'], poly_features['km_ls'], poly_features['km_labels']
    plot_KMeans(axs[2, 4], contour_reshape, km_cluster_centers, km_line_segments, km_labels)

    # Adjust the spacing of the subplots
    plt.subplots_adjust(left=0.2, top=0.9)

    # Define the row titles
    row_titles = ['One-Dimensional \nShape Features', 'Geometric Shape \nFeatures', 'Polygonal\n Approximations']

    # Set the position for the row titles
    row_title_positions = [(-1, 0.25), (-1, 0.25), (-1, 0.25)]

    # Set the row titles
    for ax, title, pos in zip(axs[:,0], row_titles, row_title_positions):
        fig.text(pos[0], pos[1], title, transform=ax.transAxes, ha='center', fontsize=10, color='blue', weight='bold', style='italic')

    # Save features for the largest contour of every image
    plot_filename = os.path.splitext(filename)[0] + "_feature_map.jpg"
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.close(fig)


# Overall process to extract features
def process_file(file_path, label, args):
    all_features = []

    if label == 's':
        contours, binary_image = preprocess_image(file_path)
        binary_images = {1: binary_image}  # Single class, use label 1
    else:
        contours, binary_images = preprocess_nii_image(file_path)

    print(f'Found {len(contours)} contours in {file_path}')

    if len(contours) == 0:
        print(f'No contours found in {file_path}')
        return

    largest_contour = max(contours, key=lambda x: cv2.contourArea(x[0] if label == 'm' else x))
    contour_number = 1 

    for i, item in enumerate(contours):
        contour, lbl = item if label == 'm' else (item, None)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue

        binary_image = binary_images[lbl] if label == 'm' else binary_image

        # Save binary_image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_filename = temp_file.name
            cv2.imwrite(temp_filename, binary_image)

        one_d_features = get_one_dimensional_features(contour, cx, cy)
        geom_features = get_geometric_shape_features(contour, cx, cy, temp_filename)  # Pass temp file path
        poly_features = get_polyognal_shape_features(contour, cx, cy, temp_filename)  # Pass temp file path

        combined_features = {'image_name': os.path.basename(file_path), 'contour_number': contour_number, 'label': lbl if lbl is not None else 1}
        combined_features.update({**one_d_features, **geom_features, **poly_features})
        all_features.append(combined_features)
        contour_number += 1 

        if np.array_equal(contour, largest_contour[0] if label == 'm' else largest_contour):
            largest_one_d_features = one_d_features
            largest_geom_features = geom_features
            largest_poly_features = poly_features

    plot_features(binary_image, largest_contour[0] if label == 'm' else largest_contour, largest_one_d_features, largest_geom_features, largest_poly_features, os.path.basename(file_path), args.output, label=largest_contour[1] if label == 'm' else None)
    print(f'[PROCESSED] {file_path}')

    if all_features:
        features_df = pd.DataFrame(all_features)
        csv_filename = os.path.splitext(os.path.basename(file_path))[0] + "_features.csv"
        features_df.to_csv(os.path.join(args.csv_save, csv_filename), index=False)
        print(f'[PROCESSED] Features saved to {os.path.join(args.csv_save, csv_filename)}')


def main():
    parser = argparse.ArgumentParser(description='Process binary mask images and extract shape features.')
    parser.add_argument('--input', type=str, required=True, help='Path to the folder containing binary mask images.')
    parser.add_argument('--csv_save', type=str, required=True, help='Folder to save the output CSV files.')
    parser.add_argument('--output', type=str, required=True, help='Folder to save the output plots.')
    parser.add_argument('--label', type=str, choices=['s', 'm'], required=True, help='Specify "s" for single class or "m" for multi-class masks.')
    parser.add_argument('--nifti_folder', type=str, help='Folder containing NIfTI files for multi-class masks (required if --label is "m").')

    args = parser.parse_args()

    if args.label == 'm' and not args.nifti_folder:
        parser.error('--nifti_folder is required when --label is "m"')

    if not os.path.exists(args.csv_save):
        os.makedirs(args.csv_save)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.label == 's':
        # Single-class case
        input_folder = args.input
        for filename in os.listdir(input_folder):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".tiff", ".tif")):
                file_path = os.path.join(input_folder, filename)
                process_file(file_path, args.label, args)
            else:
                print(f'Skipping file: {filename}')
    else:
        # Multi-class case
        if os.path.isdir(args.nifti_folder):
            for filename in os.listdir(args.nifti_folder):
                if filename.lower().endswith(".nii"):
                    file_path = os.path.join(args.nifti_folder, filename)
                    process_file(file_path, args.label, args)
                else:
                    print(f'Skipping file: {filename}')
        else:
            print(f'Invalid folder: {args.nifti_folder}')

if __name__ == "__main__":
    main()
