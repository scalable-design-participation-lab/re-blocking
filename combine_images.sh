#!/bin/bash

# Set the paths to your input folders
folderA="img/buildings"
folderB="img/parcels"

# Set the output folder
outputFolder="img/combined"
# Create the output folder if it doesn't exist
mkdir -p "$outputFolder"

# Get a list of image files in folderA
imageFilesA=("$folderA"/building_*.jpg)

# Iterate through the image files
for imageFileA in "${imageFilesA[@]}"; do
    # Get the corresponding image file in folderB
    imageName=$(basename "$imageFileA")
    imageIndex="${imageName#*_}"
    imageFileB="$folderB/parcel_$imageIndex"

    # Determine the output filename
    outputFileName="combined_$imageIndex.jpg"
    outputPath="$outputFolder/$outputFileName"

    # Stitch the images side by side using ImageMagick
    convert "$imageFileA" "$imageFileB" +append "$outputPath"

    echo "Stitched $imageFileA and $imageFileB to $outputPath"
done

echo "Image stitching complete!"
