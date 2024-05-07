#!/bin/bash

# Directory containing your input files
DIR="/data/nandini/vocab_adapt/codes/fast_align_data_new/"

# Loop through each .txt file in the directory
for filepath in "$DIR"*; do
    # Extract just the filename without extension
    filename=$(basename -- "$filepath")
    basename="${filename: -8}"

    # Run FastAlign and save the output with a new name
    ./fast_align -i "$filepath" -d -o -v -r > "${basename}_reverse.align"
    echo "Processed $filepath and saved alignment to ${basename}_reverse.align"
done
