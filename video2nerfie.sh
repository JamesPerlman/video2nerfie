#!/bin/bash

MODE=nerfie

while getopts ":i:o:m:" opt; do
    case $opt in
        i)
            INPUT_PATH=$OPTARG
            ;;
        o)
            OUTPUT_PATH=$OPTARG
            ;;
        m)
            MODE=$OPTARG
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

if [ $MODE != "hypernerf" ]; then
    MODE=nerfie
fi

COLMAP_DATA_DIR=${OUTPUT_PATH}/colmap_data
NERFIE_DATA_DIR=${OUTPUT_PATH}/nerfie_dataset
NERFIE_TRAIN_DIR=${OUTPUT_PATH}/nerfie_trained

python video2colmap/video2colmap.py -i ${INPUT_PATH} -o ${COLMAP_DATA_DIR}
python colmap2nerfie/colmap2nerfie.py -i ${COLMAP_DATA_DIR} -o ${NERFIE_DATA_DIR}
python train_${MODE}.py -i ${NERFIE_DATA_DIR} -o ${NERFIE_TRAIN_DIR}
