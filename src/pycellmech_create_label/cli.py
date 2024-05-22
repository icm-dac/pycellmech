import os
import argparse
from .main import main as create_labels_main

def main():
    parser = argparse.ArgumentParser(description='Process mask images and generate labeled images and CSV files.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing mask images.')
    parser.add_argument('--output_csv_folder', type=str, required=True, help='Path to the folder where CSV files will be saved.')
    parser.add_argument('--output_image_folder', type=str, required=True, help='Path to the folder where labeled images will be saved.')

    args = parser.parse_args()

    # Debugging output to ensure arguments are parsed correctly
    print(f"Input folder: {args.input_folder}")
    print(f"Output CSV folder: {args.output_csv_folder}")
    print(f"Output image folder: {args.output_image_folder}")
    
    create_labels_main(args.input_folder, args.output_csv_folder, args.output_image_folder)

if __name__ == "__main__":
    main()
