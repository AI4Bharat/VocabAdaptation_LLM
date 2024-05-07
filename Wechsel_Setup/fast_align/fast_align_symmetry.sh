#!/bin/bash

# Loop through each forward alignment file in the current directory
for forward_file in *_forward.align; do
    # Extract the base name for matching reverse file
    base_name="${forward_file%_forward.align}"

    # Define the corresponding reverse alignment file
    reverse_file="${base_name}_reverse.align"

    # Check if the corresponding reverse alignment file exists
    if [ -f "$reverse_file" ]; then
        # Run atools to get the symmetrical alignment and save it to a new file
        ./atools -i "$forward_file" -j "$reverse_file" -c grow-diag-final-and > "${base_name}_symmetric.align"
        echo "Generated symmetric alignment for $base_name and saved to ${base_name}_symmetric.align"
    else
        echo "Reverse file for $base_name does not exist."
    fi
done
