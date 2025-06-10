#!/bin/bash 

PATTERN=$1
TARGET=$2
OUT=$3

mkdir -p $OUT

for IMG in $PATTERN; do
    if [ "$IMG" != "$TARGET" ]; then
        filename=$(basename -- "$IMG")
        filename="${filename%.*}"
        mkdir -p $OUT/$filename
        # You may want to run this on a cluster with multiple threads
        antsRegistrationSyN.sh -d 3 -f $IMG -m $TARGET -n 8 -o $OUT/$filename/ 
    fi;
done;
