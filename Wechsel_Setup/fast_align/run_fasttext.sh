#!/bin/bash

# Directory containing your input files
DIR="/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/fastAlign_BPCC_tok/"

# Loop through each file in the directory
for filepath in "$DIR"*; do
    # Extract just the filename without extension
    filename=$(basename -- "$filepath")
    basename="${filename: -8}"

    # Run FastAlign and save the output with a new name
    ./fast_align -i "$filepath" -d -o -v > "${basename}.align"
    echo "Processed $filepath and saved alignment to ${basename}.align"
done
