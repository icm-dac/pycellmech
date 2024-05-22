import os
import argparse
from .main import main as create_nifti_main

def main():
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
    
    create_nifti_main(args.folder_path, args.input_csv_folder, args.nifti_save_dir, args.label_save_dir)

if __name__ == "__main__":
    main()
