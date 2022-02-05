#!/bin/bash

while getopts ":i:" opt; do
    case $opt in
        i)
            INPUT_PATH=$OPTARG
            ;;
    esac
done

if [ -z "${INPUT_PATH}" ]; then
    echo "Missing input path for -i argument"
    exit 1
fi

WORKDIR=/usr/local/video2nerfie

docker run \
    -it \
    --rm \
    --gpus=all \
    -v "${INPUT_PATH}":"${WORKDIR}/content" \
    video2nerfie