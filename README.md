# Denoising
Collection of utilities for denoising MRI multichannel data.

## Install
Recommended and tested is to clone this repository and use conda to create environment, using a custom name, via:

`conda create -n <env_name> -f environment.yml`

`conda activate <env_name>`

## MP-PCA based
The MP-PCA based script is based on Does et al. 2019 (https://doi.org/10.1002/mrm.27658).

#### Usage
Navigate to the "src" folder on your device and run to use with config and data from "examples/" folder: 

`python -m d_mppca`

To get an overview of the input parameters and how to use CLI inputs to change parameters of the script run:

`python -m d_mppca --help`


If used alike the examples configuration a background noise estimation is used based on St-Jean et al. 2020 (https://doi.org/10.1016/j.media.2020.101758)
which outputs a mask for background noise voxels. Those are used to estimate noise. The fit for the noise distribution is output as plot.
Denoised and De-biased maps are output.