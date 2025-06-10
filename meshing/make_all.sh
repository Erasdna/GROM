#!/bin/bash -l

PATTERN=$1
SAVE_FOLDER=$2
BRAINMASK_FOLDER=$3

mkdir -p $SAVE_FOLDER

for IMG in $PATTERN; do
    TMP=$(basename -- "$IMG")
    NAME=${TMP%-aseg.*}
    mkdir -p $SAVE_FOLDER/$NAME

    # Remove cerebellum, brain stem, 4th ventricle and optic chiasm
    mri_binarize --i $IMG \
                 --replace 8 0 \
                 --replace 7 0 \
                 --replace 15 0 \
                 --replace 16 0 \
                 --replace 46 0 \
                 --replace 47 0 \
                 --replace 85 0 \
                 --fill-holes \
                 --remove-islands \
                 --o $SAVE_FOLDER/$NAME/$NAME"_rm.mgz"
    
    mri_binarize --i $SAVE_FOLDER/$NAME/$NAME"_rm.mgz" --min 0.1 \
                 --surf-smooth 3 --surf $SAVE_FOLDER/$NAME/$NAME"_pial.stl" 

    # Make lh surface
    mri_binarize --i $SAVE_FOLDER/$NAME/$NAME"_rm.mgz" \
                 --min 0.1 --max 40 \
                 --remove-islands \
                 --fill-holes \
                 --o $SAVE_FOLDER/$NAME/"tmp_lh_"$NAME".mgz"

    mri_volcluster --in $SAVE_FOLDER/$NAME/"tmp_lh_"$NAME".mgz" \
			--thmin 1 \
			--minsize 5e4 \
			--ocn $SAVE_FOLDER/$NAME/"tmp_lh_ocn_"$NAME".mgz"

    mri_binarize --i $SAVE_FOLDER/$NAME/"tmp_lh_ocn_"$NAME".mgz" \
                --match 1 \
                --o $SAVE_FOLDER/$NAME/"tmp_lh_"$NAME".mgz"

    mri_morphology $SAVE_FOLDER/$NAME/"tmp_lh_"$NAME".mgz" \
                close $num_closing $TMP

    mri_binarize --i $SAVE_FOLDER/$NAME/"tmp_lh_"$NAME".mgz" \
                --match 1 \
                --surf-smooth 3 --surf $SAVE_FOLDER/$NAME/$NAME"_lh.stl" \
                --o $SAVE_FOLDER/$NAME/$NAME"_rm_lh.mgz"

    # Make rh surface
    mri_binarize --i $SAVE_FOLDER/$NAME/$NAME"_rm.mgz" \
                 --min 40.1 \
                 --remove-islands \
                 --fill-holes \
                 --o $SAVE_FOLDER/$NAME/"tmp_rh_"$NAME".mgz"
    
    mri_volcluster --in $SAVE_FOLDER/$NAME/"tmp_rh_"$NAME".mgz" \
			--thmin 1 \
			--minsize 5e4 \
			--ocn $SAVE_FOLDER/$NAME/"tmp_rh_ocn_"$NAME".mgz"

    mri_binarize --i $SAVE_FOLDER/$NAME/"tmp_rh_ocn_"$NAME".mgz" \
                --match 1 \
                --o $SAVE_FOLDER/$NAME/"tmp_rh_"$NAME".mgz"

    mri_morphology $SAVE_FOLDER/$NAME/"tmp_rh_"$NAME".mgz" \
                close $num_closing $TMP

    mri_binarize --i $SAVE_FOLDER/$NAME/"tmp_rh_"$NAME".mgz" \
                --match 1 \
                --surf-smooth 3 --surf $SAVE_FOLDER/$NAME/$NAME"_rh.stl" \
                --o $SAVE_FOLDER/$NAME/$NAME"_rh.mgz"
    
    # Make ventricle surface
    sh Meshing/extract-ventricles.sh $SAVE_FOLDER/$NAME/$NAME"_rm_first.mgz" $SAVE_FOLDER/$NAME/ 5 $NAME"_ventricles"

    # Apply to brainmask 
    mri_binarize --i $SAVE_FOLDER/$NAME/$NAME"_rm.mgz" \
                 --min 0.1 \
                 --o $SAVE_FOLDER/$NAME/$NAME"_rm_mask.mgz"

    mri_mask -bb 200 $BRAINMASK_FOLDER/$NAME"-brainmask.mgz" $SAVE_FOLDER/$NAME/$NAME"_rm_mask.mgz" $SAVE_FOLDER/$NAME/$NAME"_brainmask.mgz"

    mri_binarize --i $SAVE_FOLDER/$NAME/$NAME"_rm.mgz" \
                 --ventricles \
                 --match 0 \
                 --binval 0 \
                 --binvalnot 1 \
                 --o $SAVE_FOLDER/$NAME/$NAME"_rm_ventricles.mgz"

    rm $SAVE_FOLDER/$NAME/tmp*.mgz
    rm $SAVE_FOLDER/$NAME/tmp*.lut
done;