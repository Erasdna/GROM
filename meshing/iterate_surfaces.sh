#!/bin/bash 

PATTERN=$1
SAVE_DIR=$2 
MIN=$3 
MAX=$4

for SEG in $PATTERN; do 
    TMP=$(basename -- "$SEG")
    MASK=${TMP%.*}
    echo $MASK
    sh Meshing/make_stl_surface.sh $SEG $SAVE_DIR 5 $MASK $MIN $MAX
done;