#!/bin/bash

# Render all scenes
manim -pqh bp_animation.py TitleSlide
manim -pqh bp_animation.py FourSteps
manim -pqh bp_animation.py ForwardPass
manim -pqh bp_animation.py ErrorCalculation
manim -pqh bp_animation.py BackwardPass
manim -pqh bp_animation.py WeightUpdate
manim -pqh bp_animation.py Conclusion

# Create list of videos to concatenate
echo "file 'media/videos/bp_animation/1080p60/TitleSlide.mp4'" > file_list.txt
echo "file 'media/videos/bp_animation/1080p60/FourSteps.mp4'" >> file_list.txt
echo "file 'media/videos/bp_animation/1080p60/ForwardPass.mp4'" >> file_list.txt
echo "file 'media/videos/bp_animation/1080p60/ErrorCalculation.mp4'" >> file_list.txt
echo "file 'media/videos/bp_animation/1080p60/BackwardPass.mp4'" >> file_list.txt
echo "file 'media/videos/bp_animation/1080p60/WeightUpdate.mp4'" >> file_list.txt
echo "file 'media/videos/bp_animation/1080p60/Conclusion.mp4'" >> file_list.txt

# Concatenate videos
ffmpeg -f concat -safe 0 -i file_list.txt -c copy output.mp4

# Add 3 second pause between scenes
ffmpeg -i output.mp4 -filter_complex \
"[0]split=7[in1][in2][in3][in4][in5][in6][in7]; \
[in1]trim=0:7,setpts=PTS-STARTPTS[part1]; \
[in2]trim=7:14,setpts=PTS-STARTPTS[part2]; \
[in3]trim=14:24,setpts=PTS-STARTPTS[part3]; \
[in4]trim=24:32,setpts=PTS-STARTPTS[part4]; \
[in5]trim=32:42,setpts=PTS-STARTPTS[part5]; \
[in6]trim=42:50,setpts=PTS-STARTPTS[part6]; \
[in7]trim=50:57,setpts=PTS-STARTPTS[part7]; \
[part1][part2][part3][part4][part5][part6][part7]concat=n=7:v=1:a=1[outv][outa]" \
-map "[outv]" -map "[outa]" final_output.mp4

echo "Video combination complete. Final output: final_output.mp4"
