import os
from pathlib import Path
import re
import subprocess as sp

def get_fps(input_video_path: Path) -> str:
    return sp.getoutput(f" \
        ffprobe -v 0 \
            -of csv=p=0 \
            -select_streams v:0 \
            -show_entries stream=r_frame_rate \
        \"{input_video_path}\" \
    ")

def get_frame_count(input_video_path: Path) -> int:
    frames_str = sp.getoutput(f" \
        ffprobe -v error \
            -select_streams v:0 \
            -count_frames \
            -show_entries stream=nb_read_frames \
            -print_format default=nokey=1:noprint_wrappers=1 \
            {input_video_path} \
        ")
    return int(frames_str)

def extract_frames(
    input_video_path: Path,
    output_frames_path: Path,
    crop: str = "in_w:in_h:0:0",
    img_format: str = "png",
    size: str = "-1:-1",
    skip_frames: int = 15,
):
    if skip_frames < 1:
        skip_frames = 1
    
    input_video_fps = get_fps(input_video_path)

    output_frames_path.mkdir(parents=True, exist_ok=True)

    os.system(f" \
        ffmpeg \
            -i \"{input_video_path}\" \
            -r \"{input_video_fps}\" \
            -vsync vfr \
            -vf \"\
                crop={crop}, \
                scale={size}, \
                select=not(mod(n\,{skip_frames})), \
                mpdecimate, \
                setpts=N/FRAME_RATE/TB \
            \" \
            \"{output_frames_path}/%06d.{img_format}\" \
    ")
