import logging
import numpy as np
import pathlib as plib
import natsort
import nibabel as nib
from d_mmncc import distributions
log_module = logging.getLogger(__name__)


def extract_chi_noise_characteristics_from_nii(nii_data: np.ndarray,
                                               corner_fraction: float = 8.0,
                                               mask: str = "") -> (distributions.NcChi, np.ndarray):
    """
    Input slice dim or 3d nii file with sample free corners, aka noise
    dimensions assumed: [x, y, z]

    :param mask: filename for custom mask file, provided as .nii or .npy
    :param visualize: show plots of masking and extraction
    :param cmap: colormap in case of visualize is true
    :param corner_fraction: fraction of the x/y dimension corners to use for noise handling
    :param nii_data: path to nii file
    :return:
    """
    shape = nii_data.shape
    # init mask
    mask_array = np.zeros_like(nii_data, dtype=bool)
    # check for mask input
    if mask:
        mask_path = plib.Path(mask).absolute()
        if mask_path.suffix == ".nii":
            mask_array = nib.load(mask_path).get_fdata()
        elif mask_path.suffix == ".npy":
            mask_array = np.load(str(mask_path))
        else:
            log_module.error("mask file ending not recognized: "
                            "give file as .nii or .npy")
            exit(-1)
    else:
        mid_x = int(shape[0] / 2)
        mid_y = int(shape[1] / 2)
        size_x = int(shape[0] / corner_fraction)
        size_y = int(shape[1] / corner_fraction)
        # fill middle rectangle and use fftshift to shift to corners
        mask_array[mid_x - size_x:mid_x + size_x, mid_y - size_y:mid_y + size_y] = True
        mask_array = np.fft.fftshift(mask_array)
    # cast shapes to 3D
    if mask_array.shape.__len__() < 3:
        mask_array = mask_array[:, :, np.newaxis]
    if mask_array.shape.__len__() > 3:
        mask_array = np.reshape(mask_array, [*mask_array.shape[:2], -1])
    if shape.__len__() < 3:
        if shape.__len__() < 2:
            log_module.error("input data dimension < 2, input at least data slice nii")
            exit(-1)
        # input z dimension
        nii_data = nii_data[:, :, np.newaxis]
    if shape.__len__() > 3:
        # flatten time dimension onto z
        nii_data = np.reshape(nii_data, [*shape[:2], -1])

    # pick noise data from .nii, cast to 1D
    noise_array = nii_data[mask_array].flatten()
    noise_array = noise_array[noise_array > 0]

    # init distribution class object
    dist_nc_chi = distributions.NcChi()
    # emc_fit -> updates channels and sigma of ncchi object
    dist_nc_chi.fit_noise(noise_array)

    log_module.info(f"found distribution characteristics")
    dist_nc_chi.get_stats()
    # reshape to original
    if shape.__len__() > 3:
        # if 4d take maximum along time dimension, assumed to be the last
        data_max = np.max(nii_data.reshape(shape), axis=-1, keepdims=True)
        # collapse single axis
        if data_max.shape.__len__() > 4:
            data_max = data_max[:, :, :, :, 0]
        else:
            data_max = data_max[:, :, :, 0]
    else:
        # if 3d we take the total data maximum for snr mapping
        data_max = np.max(nii_data, keepdims=True)
    snr_map = np.divide(data_max, dist_nc_chi.mean(0))
    return dist_nc_chi, snr_map, noise_array


def get_noise_stats_across_slices(data_nii: np.ndarray,
                                  dim_z: int = -2, dim_t: int = -1, num_cores_mp: int = 16):
    """
    compute noise statistics using autodmri, identifying voxels belonging to noise distribution.
    advisable to check the output (especially mask) visually.
    compute the noise distribution across slices. fails if there are not enough sample free voxels within a slice
    (eg. slab selective acquisitions with tight fov).

    inputs
    nii_data: torch tensor assumed to be 4D: [x, y, z, t]
    fit_config: FitConfig configuration object for the fitting
    dim_z: int (default -2) slice dimension
    dim_t: int (default -1) time dimension
    num_cores_mp: number of cores for multiprocessing
    """
    # take first echo
    data_input = np.moveaxis(data_nii, dim_t, 0)[0]
    # if slice dimension was after time dim, we need to account for this
    if dim_z > dim_t:
        dim_z -= 1
    # if slice dimension was counted from back we need to up it
    if dim_z < 0:
        dim_z += 1

    # for now only take slab axis one echo
    s, n, m = ade.estimate_from_dwis(
        data=data_input.numpy(force=True), axis=dim_z, return_mask=True,
        exclude_mask=None, ncores=num_cores_mp, method="moments", verbose=2, fast_median=False
    )

    if fit_config.visualize:
        plotting.plot_img(m, fit_config=fit_config, name="autodmri_mask")
        plotting.plot_noise_sigma_n(sigma=s, n=n, fit_config=fit_config, name="autodmri_sigma_n")

    # assign values [we get only dim_z axis dimension]
    sigma = torch.from_numpy(s)
    num_channels = torch.from_numpy(n)
    mask = torch.from_numpy(m)

    return sigma, num_channels, mask


if __name__ == "__main__":
    path = plib.Path("./").absolute().parent
    inputFolder = './data/test_subj_data'
    path = path.joinpath(inputFolder)
    files = natsort.natsorted([path.parent.joinpath(f) for f in list(path.iterdir()) if f.suffix == ".nii"])
    data = np.array([nib.load(f).get_fdata() for f in files])
    data = np.moveaxis(data, 0, -1)

    # init mask
    mask_img = np.zeros_like(data, dtype=bool)
    corner_idx = int(mask_img.shape[0] / 8)
    mask_img[:corner_idx, -corner_idx:, :, :] = True
    mask_img[-corner_idx:, -corner_idx:, :, :] = True
    np.save("temp_mask_file.npy", mask_img)

    nc_chi, snr_data = extract_chi_noise_characteristics_from_nii(data, mask="temp_mask_file.npy")
    zero_mean = nc_chi.mean(0)
