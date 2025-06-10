# Geometry Reduced Order Modeling (GROM)

Code for the paper **"Geometry Reduced Order Modeling (GROM) with application to modeling of glymphatic function"** 

Preprint available from:<br> 
Work pending review

**Abstract**: 
> Computational modeling of the brain has become a key part of understanding how the brain clears metabolic waste, but patient-specific modeling on a significant scale is still out of reach with current methods. We introduce a novel approach for leveraging model order reduction techniques in computational models of brain geometries to alleviate computational costs involved in numerical simulations. Using image registration methods based on magnetic resonance imaging, we compute inter-brain mappings which allow previously computed solutions on other geometries to be mapped on to a new geometry. We investigate this approach on two example problems typical of modeling of glymphatic function, applied to a dataset of $101$ MRI of human patients. We discuss the applicability of the method when applied to a patient with no known neurological disease, as well as a patient diagnosed with idiopathic Normal Pressure Hydrocephalus displaying significantly enlarged ventricles

The dataset used in this work is not made publicly available due to patient privacy concerns. 

# In this repo:

- The `meshing` folder contains scripts for making `stl` surfaces from FreeSurfer generated `brainmask` and `wmparc` files of MRI
- The `registration` folder contains the script for performing registration to a target subject using ANTs
- The `run` folder contains files for running high-fidelity simulations (`run_high_fidelity.py`) and mapping solutions (`run_deformation.py`) to build a basis (`make_basis.py`) to run a reduced problem (`run_reduced.py`) with various basis sizes (not optimized for performance!)
- The `src` folder contains source code for utility classes used for meshing (`Mesher.py`), running problems (`Problem.py`, `steady_MPET.py` and `threecomp.py`) as well as other utilities

# Using this repo:

The conda environment for running python code from this repo can be installed from `requirements.txt`. In addition, SVMTK (for meshing) can be installed following the instructions from [here](https://github.com/SVMTK/SVMTK) and ANTs (for registration) can be installed following the instructions from [here](https://github.com/ANTsX/ANTs). Scripts in the `meshing` folder use [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/). 

The full pipeline in this repo may not be run in the current state without dataset access. However, we hope that sharing the code from this work may be useful for other, similar, projects.