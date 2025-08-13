#!/bin/bash


# Configuration
INPUT_DIR="media/videos/cnn_visualization/1080p60/partial_movie_files/CNNExplained"
OUTPUT_DIR="final_video"
OUTPUT_FILE="cnn_visualization_complete.mp4"
LOG_FILE="combine.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary file list
LIST_FILE="$OUTPUT_DIR/filelist.txt"
> "$LIST_FILE"

echo "=== Video Combination Process Started ===" | tee "$LOG_FILE"
echo "Scanning for partial video files..." | tee -a "$LOG_FILE"

# Find and sort files by their sequence numbers
find "$INPUT_DIR" -name "*.mp4" -type f | sort -t '_' -n -k1 | while read -r file; do
    if [[ -f "$file" ]]; then
        echo "file '$PWD/$file'" >> "$LIST_FILE"
        echo "Added: $file" | tee -a "$LOG_FILE"
    else
        echo "Warning: File not found - $file" | tee -a "$LOG_FILE"
    fi
done

echo "Found $(wc -l < "$LIST_FILE") video chunks to combine" | tee -a "$LOG_FILE"

# Combine videos using ffmpeg
echo "Combining videos..." | tee -a "$LOG_FILE"
ffmpeg -f concat -safe 0 -i "$LIST_FILE" -c copy "$OUTPUT_DIR/$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"

# Verify output
if [[ -f "$OUTPUT_DIR/$OUTPUT_FILE" ]]; then
    echo "Successfully created: $OUTPUT_DIR/$OUTPUT_FILE" | tee -a "$LOG_FILE"
    echo "File size: $(du -h "$OUTPUT_DIR/$OUTPUT_FILE" | cut -f1)" | tee -a "$LOG_FILE"
    # Clean up
    rm "$LIST_FILE"
else
    echo "Error: Failed to create output video" | tee -a "$LOG_FILE"
    echo "Temporary file list kept at: $LIST_FILE for debugging" | tee -a "$LOG_FILE"
fi

echo "=== Process Complete ===" | tee -a "$LOG_FILE"
