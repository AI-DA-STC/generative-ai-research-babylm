#!/bin/bash

# Define the root directories
MAIN_DIR=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")
#dir where all the raw data is copied
RAW_DIR="$MAIN_DIR/data/raw"
#dir where each data file is preprocessed
PREPROCESSED_DIR="$MAIN_DIR/data/preprocessed"
#dir where all the .py scripts are stored for each data file 
PREPROCESS_DIR="$MAIN_DIR/preprocess"
#dir for combining all the preprocessed data files
COMBINED_DIR="$MAIN_DIR/data/processed"

# Create the preprocessed directory if it doesn't exist
mkdir -p "$PREPROCESSED_DIR"
# Function to process a single file
process_file() {
    local input_file=$1
    local output_dir=$2
    local script_name=$(basename "$input_file" | sed 's/\.[^.]*$//')".py"
    local script_path="$PREPROCESS_DIR/$script_name"

    # Check if the corresponding Python script exists
    if [ ! -f "$script_path" ]; then
        echo "Error: No matching script found for $input_file"
        return 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    local output_file="$output_dir/$(basename "$input_file")"
    echo "Processing $input_file to $output_file using $script_name"
    python3 "$script_path" "$input_file" "$output_file"
}

# Process files for dev, train_10M, train_100M
for file in "train_10M_old" "train_100M_old" "dev_old"; do
    input_dir="$RAW_DIR/$file"
    output_dir="$PREPROCESSED_DIR/$file"
    for input_file in "$input_dir"/*.train; do
        if [ -f "$input_file" ]; then
            process_file "$input_file" "$output_dir"
        fi
    done
    for input_file in "$input_dir"/*.dev; do
        if [ -f "$input_file" ]; then
            process_file "$input_file" "$output_dir"
        fi
    done
done

echo "All processing complete."


# Combine files for each folder
echo "Combining preprocessed files..."
for folder in "train_10M_old" "train_100M_old" "dev_old"; do
    input_dir="$RAW_DIR/$folder"
    output_dir="$COMBINED_DIR/$folder"
    
    echo "Combining files in $folder"
    python3 "$PREPROCESS_DIR/combine.py" "$input_dir" "$output_dir"
done

echo "All combining complete."