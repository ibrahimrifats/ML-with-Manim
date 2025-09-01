#!/bin/bash

# Configuration
SCENES=("MLPTitle" "NeuronImageScene" "MLPAnimation" "FinalSummary")
OUTPUT_DIR="rendered_scenes"
FINAL_OUTPUT="mlp_deep_dive_final.mp4"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Render each scene
echo "Starting video rendering..."
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    output_file="$OUTPUT_DIR/temp_part_$((i+1)).mp4"
    temp_name="temp_part_$((i+1))"

    echo "Rendering $scene..."

    # Render the scene and specify output filename
    manim -ql -o "$temp_name" ad.py "$scene"

    # Move the rendered file to the output directory
    rendered_path="media/videos/ad/1920p30/${temp_name}.mp4"
    if [ -f "$rendered_path" ]; then
        mv "$rendered_path" "$output_file"
    else
        echo "Error: Failed to render $scene"
        exit 1
    fi
done

# Combine videos
echo "Combining videos..."
(
    cd "$OUTPUT_DIR" || exit 1
    echo "Creating file list..."
    > input.txt
    for f in temp_part_*.mp4; do
        echo "file '$f'" >> input.txt
    done

    ffmpeg -f concat -safe 0 -i input.txt -c copy "$FINAL_OUTPUT"
    mv "$FINAL_OUTPUT" ..
)

# Verify final output
if [ ! -f "$FINAL_OUTPUT" ]; then
    echo "Error: Failed to create final video"
    exit 1
fi

# Cleanup
echo "Cleaning up..."
rm -rf "$OUTPUT_DIR"

echo "✅ Success! Final video created: $FINAL_OUTPUT"
