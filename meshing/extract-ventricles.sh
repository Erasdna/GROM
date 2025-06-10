#!/bin/bash

INPUT=$1
OUTPUT_PATH=$2
SMOOTHING=$3
NAME=$4

mkdir -p $OUTPUT_PATH
OUTPUT_VOL=$OUTPUT_PATH/$NAME.mgz
OUTPUT_SURF=$OUTPUT_PATH/$NAME.stl
TMP=$OUTPUT_PATH/tmp.mgz
TMP_OCN=$OUTPUT_PATH/tmp-ocn.mgz

num_closing=2
V_min=100

#if [ "$postprocess" == true ]; then
mri_binarize --i $INPUT --ventricles \
			--o $TMP

mri_volcluster --in $TMP \
			--thmin 1 \
			--minsize $V_min \
			--ocn $TMP_OCN

mri_binarize --i $TMP_OCN \
			--match 1 \
			--o $TMP

mri_morphology $TMP \
			close $num_closing $TMP

mri_binarize --i $TMP \
			--match 1 \
			--surf-smooth $SMOOTHING \
			--surf $OUTPUT_SURF \
			--o $OUTPUT_VOL

rm $TMP
rm $TMP_OCN
