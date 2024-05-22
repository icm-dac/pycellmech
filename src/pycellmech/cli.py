import os
import argparse
from .main import process_file

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
