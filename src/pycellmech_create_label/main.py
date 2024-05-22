"""
File: <pycellmech_create_label.py>

Author: Janan ARSLAN (janan.arslan@icm-institute.org)
Institut du Cerveau - Data Analysis Core (DAC) Platform
Created: 22 FEB 2024
Last Modified: 22 MAY 2024
Modification Author: Janan ARSLAN

To assist with multi-class processing, this code allows the end-user to first
create a CSV file by identifying connected regions and assigning a numerical label
accordingly. The output is a CSV file with cell_id (region of interest), centroid_x
and centroid_y (the centroid coordinates of the region), and an empty column with the
header final_label.

Users can populated the final_label and create a NifTI file using pycellmech_nifti. 

"""


import cv2
import numpy as np
import os
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def main(input_folder, output_csv_folder, output_image_folder):
    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            mask_path = os.path.join(input_folder, filename)
            print(f"Processing file: {mask_path}")  # Debugging output
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                print(f"Failed to load image: {mask_path}")  # Debugging output
                continue  
            
            binary_mask = (mask > 0).astype(int)
            labeled_mask = label(binary_mask, connectivity=1)  # Label connected regions
            labeled_rgb = label2rgb(labeled_mask, bg_label=0)  # Convert labeled mask to RGB for visualization
            image_to_draw = (labeled_rgb * 255).astype(np.uint8)  

            # Get properties of each region to find centroids
            regions = regionprops(labeled_mask)
            
            # Put labels on each cell
            for region in regions:
                y, x = map(int, region.centroid)
                cv2.putText(image_to_draw, str(region.label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (255, 255, 255), 2)

            labels, counts = np.unique(labeled_mask, return_counts=True)  # Extract labels and counts
            df = pd.DataFrame({
                'cell_id': labels[1:],  # Exclude the background label
                'centroid_x': [int(region.centroid[1]) for region in regions],
                'centroid_y': [int(region.centroid[0]) for region in regions],
                'final_label': [''] * (len(labels) - 1) # Empty label for end-user to complete
            })

            base_filename = os.path.splitext(filename)[0]
            csv_output_path = os.path.join(output_csv_folder, f'{base_filename}.csv')
            image_output_path = os.path.join(output_image_folder, f'{base_filename}.png')

            # Save CSV file
            df.to_csv(csv_output_path, index=False)
            print(f"Saved CSV file to: {csv_output_path}")  # Debugging output

            # Save the labeled image
            cv2.imwrite(image_output_path, image_to_draw)
            print(f"Saved labeled image to: {image_output_path}")  # Debugging output

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process mask images and generate labeled images and CSV files.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing mask images.')
    parser.add_argument('--output_csv_folder', type=str, help='Path to the folder where CSV files will be saved.')
    parser.add_argument('--output_image_folder', type=str, help='Path to the folder where labeled images will be saved.')

    args = parser.parse_args()
    
    # Debugging output to ensure arguments are parsed correctly
    print(f"Input folder: {args.input_folder}")
    print(f"Output CSV folder: {args.output_csv_folder}")
    print(f"Output image folder: {args.output_image_folder}")
    
    main(args.input_folder, args.output_csv_folder, args.output_image_folder)
