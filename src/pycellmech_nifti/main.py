"""
File: <pycellmech_nifti.py>

Author: Janan ARSLAN (janan.arslan@icm-institute.org)
Institut du Cerveau - Data Analysis Core (DAC) Platform
Created: 22 FEB 2024
Last Modified: 22 MAY 2024
Modification Author: Janan ARSLAN

This code allows an end-user to convert their CSV-labeled files to a NifTI format
in order to extract shape-related features for multi-class data. It updates the labeled mask,
corrects the orientation, converts it to a SimpleITK image, and saves it as a NIfTI file.

Users can input the NifTI in parallel to their binary mask for procesing in the main
pycellmech command line. 

"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from collections import Counter
import SimpleITK as sitk
import argparse

def main(folder_path, input_csv_folder, nifti_save_dir, label_save_dir):
    os.makedirs(nifti_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    for mask_filename in os.listdir(folder_path):
        if mask_filename.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif')):
            mask_path = os.path.join(folder_path, mask_filename)
            mask_basename = os.path.splitext(mask_filename)[0]

            mask = cv2.imread(mask_path, 0)
            binary_mask = (mask > 0).astype(int)
            labeled_mask = label(binary_mask, connectivity=1)

            csv_filename = f"{mask_basename}.csv"
            csv_path = os.path.join(input_csv_folder, csv_filename)
            if not os.path.exists(csv_path):
                print(f"CSV file for {mask_filename} not found, skipping.")
                continue
            final_labels_df = pd.read_csv(csv_path)

            # Create a mapping from cell_id to final_label
            label_map = dict(zip(final_labels_df['cell_id'], final_labels_df['final_label']))

            # Update the labeled mask with final labels
            final_labeled_mask = np.zeros_like(labeled_mask)
            for region in regionprops(labeled_mask):
                original_label = region.label
                final_label = label_map.get(original_label, 0)  # Default to 0 if not found
                final_labeled_mask[labeled_mask == original_label] = final_label

            # Save the updated labeled mask before rotation
            labeled_image = label2rgb(final_labeled_mask, bg_label=0)
            updated_label_image_path = os.path.join(label_save_dir, f"{mask_basename}_labeled.png")
            plt.imsave(updated_label_image_path, labeled_image)

            # Correct the orientation
            corrected_final_labeled_mask = np.rot90(final_labeled_mask, k=1)
            corrected_final_labeled_mask = np.flipud(corrected_final_labeled_mask)

            # Convert to SimpleITK image and save as NIfTI
            original_sitk_image = sitk.ReadImage(mask_path)
            corrected_final_labeled_mask_sitk = sitk.GetImageFromArray(corrected_final_labeled_mask.astype(np.int16))
            corrected_final_labeled_mask_sitk.CopyInformation(original_sitk_image)
            nifti_save_path = os.path.join(nifti_save_dir, f"{mask_basename}.nii")
            sitk.WriteImage(corrected_final_labeled_mask_sitk, nifti_save_path)

            # Calculate the unique labels in the corrected final labeled mask
            unique_labels = np.unique(corrected_final_labeled_mask)

            # Load the saved NIfTI file
            saved_image = sitk.ReadImage(nifti_save_path)
            saved_array = sitk.GetArrayFromImage(saved_image)

            # Calculate the counter on the saved NIfTI file
            saved_label_counts = Counter(saved_array.flatten())

            # Debugging: Print occurrences of each final label in the saved NIfTI file
            print(f"Occurrences of each final label in the saved NIfTI file for {mask_filename}:")
            print(saved_label_counts)
            print(f"Unique labels in corrected final labeled mask: {unique_labels}")

            # Verify that unique_labels match the final_labels in the CSV
            expected_labels = final_labels_df['final_label'].unique()
            missing_labels = set(expected_labels) - set(unique_labels)
            extra_labels = set(unique_labels) - set(expected_labels)

            if missing_labels:
                print(f"Missing labels in final labeled mask: {missing_labels}")
            else:
                print("All expected labels are present in the final labeled mask.")

            if extra_labels:
                print(f"Extra labels in final labeled mask: {extra_labels}")
            else:
                print("No extra labels in the final labeled mask.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process mask images, generate labeled images and CSV files, and save as NIfTI.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the input folder containing mask images.')
    parser.add_argument('--input_csv_folder', type=str, required=True, help='Path to the folder containing CSV files with labels.')
    parser.add_argument('--nifti_save_dir', type=str, required=True, help='Path to the folder where NIfTI files will be saved.')
    parser.add_argument('--label_save_dir', type=str, required=True, help='Path to the folder where labeled images will be saved.')

    args = parser.parse_args()
    
    # Debugging output to ensure arguments are parsed correctly
    print(f"Mask directory: {args.folder_path}")
    print(f"CSV directory: {args.input_csv_folder}")
    print(f"NIfTI save directory: {args.nifti_save_dir}")
    print(f"Label save directory: {args.label_save_dir}")
    
    main(args.folder_path, args.input_csv_folder, args.nifti_save_dir, args.label_save_dir)
