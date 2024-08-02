#!/bin/bash

# Change to your target directory
cd /work/re-blocking/data/all-cities-20k

# Initialize a counter
counter=0

# Loop over each file in the directory
for file in *; do
  # Check if it is a file (and not a directory)
  if [ -f "$file" ]; then
    counter=$((counter + 1))
    # Delete every 4th file
    if [ $((counter % 4)) -eq 0 ]; then
      echo "Deleting: $file"
      rm "$file"
    fi
  fi
done