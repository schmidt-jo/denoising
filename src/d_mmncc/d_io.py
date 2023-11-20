import typing
import nibabel as nib
import pathlib as plib
import logging
import numpy as np
log_module = logging.getLogger(__name__)


def check_name(name: str):
    if not name:
        err = "no filename given to save file to"
        log_module.error(err)
        raise AttributeError(err)


def load_nii_data(file_path_nii: str | plib.Path) -> (np.ndarray, nib.Nifti1Image):
    # ensure path is plib.Path, in case of str input
    file_path_nii = plib.Path(file_path_nii).absolute()
    # check if file
    if not file_path_nii.is_file():
        err = f"File : {file_path_nii.as_posix()} not found!"
        log_module.error(err)
        raise FileNotFoundError(err)
    if ".nii" not in file_path_nii.suffixes:
        err = f"File : {file_path_nii.as_posix()} not a .nii file."
        log_module.error(err)
        raise AttributeError(err)
    # load
    log_module.info(f"Loading Nii File: {file_path_nii.as_posix()}")
    nii_img = nib.load(file_path_nii.as_posix())
    nii_data = nii_img.get_fdata()
    return nii_data, nii_img


def save_nii(data: typing.Union[nib.Nifti1Image, np.ndarray], file_path: str | plib.Path, name: str,
             affine: np.ndarray = None):
    check_name(name=name)
    # make plib Path ifn
    file_path = plib.Path(file_path).absolute().joinpath(name).with_suffix(".nii")
    if isinstance(data, np.ndarray):
        if affine is None:
            err = "Provide Affine to save .nii"
            log_module.error(err)
            raise AttributeError(err)
        img = nib.Nifti1Image(data, affine)
    else:
        img = data
    log_module.info(f"Writing File: {file_path.as_posix()}")
    nib.save(img, file_path.as_posix())

