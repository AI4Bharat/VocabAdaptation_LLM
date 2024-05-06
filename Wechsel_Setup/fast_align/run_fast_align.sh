#!/bin/bash

# Directory containing your input files
DIR="/data/nandini/vocab_adapt/codes/fast_align_data_foreign/"

# Loop through each .txt file in the directory
for filepath in "$DIR"*; do
    # Extract just the filename without extension
    filename=$(basename -- "$filepath")
    basename="${filename: -8}_10"

    # Run FastAlign and save the output with a new name
    ./fast_align -i "$filepath" -d -o -v -I 10 > "${basename}.align"
    echo "Processed $filepath and saved alignment to ${basename}.align"
done


#this file will be run in codes/fast_align/build
