import logging
from d_mmncc import options, d_io, plotting, chambolle_pock
from autodmri import estimator as ade
import numpy as np
import multiprocessing as mp
import pathlib as plib
import scipy.special as ssp
import tqdm


def main(config: options.Config):
    # setup
    path = plib.Path(config.save_path).absolute()
    file_name = plib.Path(config.nii_path).absolute().stem
    chambolle_pock_lambda = 0.05
    chambolle_pock_num_iter = 25
    # calculate cores to use, minimum 4
    num_cpus = np.max([mp.cpu_count() - config.mp_headroom, 4])

    logging.info(f"__ Loading input")
    nii_data, nii_img = d_io.load_nii_data(config.nii_path)
    # input assumed to be 4d
    while len(nii_data.shape) < 4:
        nii_data = np.expand_dims(nii_data, -1)
    # rescale if range low
    if np.max(nii_data) < 10:
        logging.info(f"rescaling data to avoid float / calc errors")
        rescale_factor = np.max(nii_data) / 1000
        nii_data /= rescale_factor
    else:
        rescale_factor = 1.0
    # run denoizing
    logging.info(f"__ Run")
    # for now take first echo to extract mask
    d_nii_data = nii_data[:, :, :, 0].copy()
    # extract noise
    logging.info("__ Extract noise stats")
    sigma, num_channels, mask = ade.estimate_from_dwis(
        data=d_nii_data, axis=-1, return_mask=True,
        exclude_mask=None, ncores=num_cpus, method="moments", verbose=2, fast_median=False
    )
    # save for debugging
    # reshape
    mask = np.repeat(mask[:, :, :, None], axis=-1, repeats=nii_data.shape[-1])
    # save mask
    d_io.save_nii(mask, file_path=path, name=f"autodmri_mask", affine=nii_img.affine)
    # calculate missing stats for later echoes
    for k in tqdm.trange(nii_data.shape[2], desc="get noise dist per slice"):
        # repeat extraction of noise
        noise_voxels = nii_data[:, :, k][mask[:, :, k].astype(bool)]
        noise_voxels = noise_voxels[noise_voxels > 0]
        sigma[k] = get_sigma_from_noise_vox(noise_voxel_data=noise_voxels)
        num_channels[k] = get_n_from_noise_vox(noise_voxel_data=noise_voxels, sigma=sigma[k])

    # take whole volume for denoising
    d_nii_data = nii_data.copy()
    mp_list = [(
        d_nii_data[:, :, k],
        sigma[k],
        num_channels[k],
        config.max_num_iter,
        config.solver_max_num_iter,
        config.solver_tv_lambda,
        k) for k in range(d_nii_data.shape[2])
    ]
    num_cpus = np.max([4, mp.cpu_count() - config.mp_headroom])  # take at least 4, leave mp Headroom
    logging.info(f"multiprocessing using {num_cpus} cpus")
    with mp.Pool(num_cpus) as p:
        results = list(
            tqdm.tqdm(p.imap(wrap_for_mp, mp_list), total=d_nii_data.shape[2], desc="multi-processing data"))
    for item in results:
        data, idx = item
        d_nii_data[:, :, idx] = data
    # for slice_idx in tqdm.trange(nii_data.shape[2], desc=f"processing slices"):
    #     y = d_nii_data[:, :, slice_idx].copy()
    #     x = d_nii_data[:, :, slice_idx].copy()
    #     # dims x,y [x,y,z,t], sigma [z], n [z]
    #     y = _y_tilde(y_obs=y, x_approx=x,
    #                  sigma=sigma[slice_idx], n=np.round(num_channels)[slice_idx].astype(int))
    #     d_nii_data[:, :, slice_idx] = chambolle_pock.chambolle_pock_tv(
    #         y, chambolle_pock_lambda, n_it=config.num_max_runs
    #     )
    if config.visualize:
        plotting.plot_noise_sigma_n(sigma=sigma, n=num_channels, config=config,
                                    name=f"autodmri_sigma_n_mn-{config.max_num_iter}_"
                                         f"stvlam-{config.solver_tv_lambda:.2f}".replace(".", "p"))

    name = (f"{config.file_prefix}_{file_name}_mn-{config.max_num_iter}_"
            f"stvlam-{config.solver_tv_lambda:.2f}").replace(".", "p")
    d_io.save_nii(d_nii_data * rescale_factor, file_path=config.save_path, name=name, affine=nii_img.affine)


def wrap_for_mp(args):
    d_nii_data_slice, sig, n, max_num_iter_steps, solver_num_max_its, solver_lambda, slice_idx = args
    y = d_nii_data_slice.copy()
    x = d_nii_data_slice.copy()
    x_old = np.full_like(x, 1000)
    break_counter = 0
    # chosen max number of iterations
    for _ in range(max_num_iter_steps):
        diff_last = np.max(np.abs(x - x_old))
        x_old = x.copy()
        # dims x,y [x,y,t], sigma, n
        y = _y_tilde(y_obs=y, x_approx=x,
                     sigma=sig, n=n)
        x = chambolle_pock.chambolle_pock_tv(
            y, solver_lambda, n_it=solver_num_max_its
        )
        diff_new = np.max(np.abs(x - x_old))
        convergence = np.abs(diff_new - diff_last)
        if convergence < 1e-2:
            # iterate not changing much
            # exit if happens for 3 consecutive runs
            if break_counter > 2:
                break
            break_counter += 1
    d_nii_data_slice = x
    return d_nii_data_slice, slice_idx


def get_n_from_noise_vox(noise_voxel_data: np.ndarray, sigma: float):
    return 1 / (2 * noise_voxel_data.shape[0] * sigma ** 2) * np.sum(noise_voxel_data ** 2, axis=0)


def get_sigma_from_noise_vox(noise_voxel_data: np.ndarray):
    num_pts = noise_voxel_data.shape[0]
    return np.sqrt(1 / 2) * np.sqrt(
        np.sum(noise_voxel_data ** 4, axis=0) / np.sum(noise_voxel_data ** 2, axis=0) -
        1 / num_pts * np.sum(noise_voxel_data ** 2, axis=0)
    )


def _y_tilde(y_obs: np.ndarray, x_approx: np.ndarray, sigma: np.ndarray | float | int, n: np.ndarray | float | int):
    arg = np.multiply(y_obs, x_approx) / sigma ** 2
    factor = _majorante(arg, n)
    return y_obs * factor


def _majorante(arg_arr: np.ndarray, n: np.ndarray | float | int):
    result = np.zeros_like(arg_arr)
    gam: float = 7e2  # set from which signal size onwards we approx with gaussian behavior
    eps: float = 1e-5  # for comparing 0
    # for smaller eps result array remains 0
    # for small enough args but bigger than eps we compute the given formula
    sel = (eps < arg_arr) & (gam > arg_arr)
    result[sel] = np.divide(
        ssp.iv(n, arg_arr[sel]),
        ssp.iv(n - 1, arg_arr[sel])
    )
    # for big args we linearly approach asymptote to 1 @ input arg 30000 (random choice
    len_asymptote = 3e4
    start_val = np.divide(
        ssp.iv(n, gam),
        ssp.iv(n - 1, gam)
    )
    sel = arg_arr >= gam
    result[sel] = start_val + (1.0 - start_val) / len_asymptote * (arg_arr[sel] - gam)
    return result


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("___________________________________________________________________")
    logging.info("___ MM Denoizing using automated parameter selection (autodmri) ___")
    logging.info("___________________________________________________________________")

    parser, prog_args = options.create_cli()

    opts = options.Config.from_cli(prog_args)
    # set logging level after possible config file read
    if opts.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        main(config=opts)

    except Exception as e:
        logging.exception(e)
        parser.print_usage()
