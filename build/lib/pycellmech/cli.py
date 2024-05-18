import argparse
from .main import process_images

def main():
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing binarized images')
    parser.add_argument('--csv_file_path', type=str, required=True, help='Path to save the shape features CSV file')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save the feature maps')

    args = parser.parse_args()
    process_images(args.folder_path, args.csv_file_path, args.output_folder)

if __name__ == "__main__":
    main()
