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
    video2nerfie