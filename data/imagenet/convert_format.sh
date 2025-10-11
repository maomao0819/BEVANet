#!/bin/bash

# Function to convert .JEPG to the specified format
convert_format() {
    local dir="$1"
    local replaced_fromat="$2"
    local desired_format="$3"
    find "$dir" -type f -iname "*.$replaced_fromat" | while read -r file; do
        new_file="${file%.$replaced_fromat}.$desired_format"
        # mv "$file" "$new_file"
        if mv "$file" "$new_file"; then
            echo "Converted $file to $new_file"
        else
            echo "Failed to convert $file"
        fi
    done
}

# Usage
replaced_fromat="JEPG"
desired_format="jpeg"  # Replace with the desired format (e.g., jpeg, jpg, png)
root_directory="train"  # Replace with the path to your ImageNet dataset folder
convert_format "$root_directory" "$replaced_fromat" "$desired_format"
root_directory="val"  # Replace with the path to your ImageNet dataset folder
convert_format "$root_directory" "$replaced_fromat" "$desired_format"