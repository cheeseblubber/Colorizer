import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def move_file(item, sorted_dir, marker, num_chars):
    """Move a file to the sorted directory based on specific characters following a marker."""
    if item.is_file() and item.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif'):
        # Find the start index of the marker and calculate the start of the desired substring
        start_index = item.stem.find(marker) + len(marker)
        if start_index > len(marker) - 1:  # Ensure the marker was found
            # Extract the substring of the specified length
            substring = item.stem[start_index:start_index + num_chars]
            if substring:  # Ensure substring is not empty
                target_dir = Path(sorted_dir, substring)
                target_dir.mkdir(exist_ok=True)
                shutil.move(str(item), target_dir.joinpath(item.name))  # Changed from copy to move

def sort_images(source_dir, sorted_dir, marker, num_chars):
    """Sort images into subdirectories based on a substring following a given marker."""
    # Create the sorted directory if it doesn't exist
    Path(sorted_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all files in the source directory
    items = [item for item in Path(source_dir).iterdir()]

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        # Submit tasks to the executor
        for item in items:
            executor.submit(move_file, item, sorted_dir, marker, num_chars)

# Specify the source and destination directories
source_directory = '/home/ubuntu/workspace/sketch/'  # Change this to your source directory
sorted_directory = '/home/ubuntu/workspace/sketch-sorted'  # Change this to your sorted directory

# Marker string and number of characters to extract
marker_string = "gray-scale--"
characters_to_extract = 9

# Call the function
sort_images(source_directory, sorted_directory, marker_string, characters_to_extract)

