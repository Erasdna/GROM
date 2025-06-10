#!/bin/bash

echo "$@"

INPUT=$1
OUTPUT_PATH=$2
SMOOTHING=$3
NAME=$4
MIN=$5
MAX=$6

mkdir -p $OUTPUT_PATH
OUTPUT_VOL=$OUTPUT_PATH/$NAME.mgz
OUTPUT_SURF=$OUTPUT_PATH/$NAME.stl

mri_binarize --i $INPUT \
            --min $MIN \
            --max $MAX \
            --surf-smooth $SMOOTHING \
            --surf $OUTPUT_SURF \
            --o $OUTPUT_VOL
