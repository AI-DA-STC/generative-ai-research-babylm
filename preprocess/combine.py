import os
import sys
from tqdm import tqdm

def combine_files(input_folder, output_folder):
    # Get list of files in the input folder
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    if not files:
        print(f"No files found in {input_folder}")
        return
    else:
        print(f"files found in {input_folder}:{files}")

    # Get the extension of the first file (assuming all files have the same extension)
    _, extension = os.path.splitext(files[0])

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Name of the combined file
    combined_filename = f"combined{extension}"
    combined_filepath = os.path.join(output_folder, combined_filename)

    # Combine files
    with open(combined_filepath, 'w', encoding='utf-8') as outfile:
        for filename in tqdm(files, desc="Combining files"):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            outfile.write('\n')  # Add a newline between files

    print(f"Combined file saved as {combined_filepath}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python combine_files.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory")
        sys.exit(1)

    combine_files(input_folder, output_folder)

if __name__ == "__main__":
    main()