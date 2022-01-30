#!/bin/bash

while getopts ":i:o:" opt; do
    case $opt in
        i)
            INPUT_PATH=$OPTARG
            ;;
        o)
            OUTPUT_PATH=$OPTARG
            ;;
    esac
done

if [ -z "${INPUT_PATH}" ]; then
    echo "Missing input path for -i argument"
    exit 1
fi

if [ -z "${OUTPUT_PATH}" ]; then
    echo "Missing output path for -o argument"
    exit 1
fi

WORKDIR=/usr/local/video2nerfie

docker run \
    -it \
    --rm \
    --gpus=all \
    -v "${INPUT_PATH}":"${WORKDIR}/content/input" \
    -v "${OUTPUT_PATH}":"${WORKDIR}/content/output" \
    -v /usr/lib/wsl:/usr/lib/wsl \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg:/mnt/wslg \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e LIBGL_ALWAYS_SOFTWARE=1 \
    --device=/dev/dxg \
    video2nerfie