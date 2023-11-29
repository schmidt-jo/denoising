"""
MPPCA denoising using method outlined in:
Does et al. 2019: Evaluation of principal component analysis image denoising on
multi‐exponential MRI relaxometry. Magn Reson Med
DOI: 10.1002/mrm.27658
_____
24.11.2023, Jochen Schmidt
"""
import logging
import pathlib as plib
import numpy as np
import tqdm
from d_mppca import options
import torch
import nibabel as nib
import plotly.express as px

logging.getLogger("simple_parsing").setLevel(logging.WARNING)


def den_patch_mp(args):
    coll = []
    for item in args:
        patch, (idx, idy, idz), (m, n_v) = item
        patch_shape = patch.shape
        patch = np.reshape(patch, (-1, m))
        # do mean removed
        patch = patch - np.mean(patch, axis=0)
        # do svd
        u, s, v = np.linalg.svd(patch, full_matrices=False, compute_uv=True)

        # calculate inequality
        left = np.array([(s[p + 1] - s[m - 1]) / (4 * np.sqrt((m - p) / n_v)) for p in range(m - 1)])
        right = np.array([(1 / (m - p)) * np.sum(s[:p + 1]) for p in range(m - 1)])
        # minimum p for which left < right
        p = np.where(left < right)[0][0]

        # calculate denoised data
        d = np.matmul(u[:, :p], np.matmul(np.diag(s[:p]), v[:p]))
        # shape back
        d = np.reshape(d, patch_shape)
        # collect
        coll.append([d, (idx, idy, idz)])
    return coll


def main(config: options.Config):
    file_path = plib.Path(config.in_path).absolute()
    if not file_path.is_file():
        err = f"file provided ({file_path.as_posix()}) not found"
        logging.error(err)
        raise FileNotFoundError(err)
    path = file_path.parent
    if config.save_path:
        save_path = plib.Path(config.save_path).absolute()
        save_path.mkdir(exist_ok=True, parents=True)
    else:
        save_path = path
    logging.info(f"load file: {file_path}")
    if ".nii" in file_path.suffixes:
        img = nib.load(file_path.as_posix())
        img_data = torch.from_numpy(img.get_fdata())
        affine = torch.from_numpy(img.affine)
    elif ".pt" in file_path.suffixes:
        img_data = torch.load(file_path.as_posix())
        af_path = plib.Path(config.in_affine).absolute()
        if not af_path.is_file():
            err = f"file provided ({af_path.as_posix()}) not found"
            logging.error(err)
            raise FileNotFoundError(err)
        affine = torch.load(af_path.as_posix())
    else:
        err = "file not .nii or .pt file"
        logging.error(err)
        raise AttributeError(err)
    name = f"{config.file_prefix}_mppca_rsos_fixed-p-{config.fixed_p}"

    if config.use_gpu:
        device = torch.device(f"cuda:{config.gpu_device}")
    else:
        device = torch.device("cpu")
    # logging.info("fft to image space")
    img_shape = img_data.shape

    # we need to batch the data to fit on memory, easiest is to do it dimension based
    # want to batch channel dim and two slice axis (or any dim)
    # get vars
    nx, ny, nz, nch, m = img_shape
    # cube_side_len = 3
    cube_side_len = torch.ceil(torch.sqrt(torch.tensor([m]))).to(torch.int).item()
    n_v = cube_side_len ** 2
    ncx = nx - cube_side_len
    ncy = ny - cube_side_len
    # calculate const for mp inequality
    m_mp = min(m, n_v)
    if config.fixed_p > 0:
        p = config.fixed_p
        left_b = None
        right_a = None
        r_cumsum = None
    else:
        p = None
        m_mp_arr = torch.arange(m_mp - 1)
        left_b = 4 * torch.sqrt(torch.tensor((m_mp - m_mp_arr) / n_v)).to(device=device, dtype=torch.float64)
        right_a = (1 / (m_mp - m_mp_arr)).to(device=device, dtype=torch.float64)
        # build a matrix to make the cummulative sum for the inequality calculation a matrix multiplication
        # dim [mmp, mmp - 1]
        r_cumsum = torch.triu(torch.ones(m_mp - 1, m_mp), diagonal=1).to(device=device, dtype=torch.float64)

    # loop over dim slices, batch dim channels
    logging.info(f"fft to image space")
    img_data = torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(
                img_data,
                dim=(0, 1)
            ),
            dim=(0, 1)
        ),
        dim=(0, 1)
    )
    if config.normalize:
        logging.info("normalize data, max 1 magnitude across time dimension")
        img_data = img_data / torch.max(torch.abs(img_data), dim=-1, keepdim=True)[0]
        name = f"{name}_normalized-input"
    img_data = torch.movedim(img_data, (2, 3), (0, 1))
    data_denoised = torch.zeros_like(img_data)
    data_access = torch.zeros(data_denoised.shape[:-1], dtype=torch.float)
    data_p = torch.zeros(data_access.shape, dtype=torch.int)
    data_p_avg = torch.zeros_like(data_p)
    logging.info(f"start processing")
    # x steps batched
    x_steps = torch.arange(ncx)[:, None] + torch.arange(cube_side_len)[None, :]
    for idx_y in tqdm.trange(img_data.shape[3] - cube_side_len, desc="loop over dim 1",
                             position=0, leave=False):
        patch = img_data[:, :, x_steps, idx_y:idx_y+cube_side_len].to(device)
        patch_shape = patch.shape
        patch = torch.reshape(patch, (nz, nch, ncx, -1, m))
        patch = torch.movedim(patch, -1, -2)
        # try batched svd
        # patch = img_data[:, :, start_x:end_x, start_y:end_y].to(device)
        # remove mean across spatial dim of patch
        patch_mean = torch.mean(patch, dim=-1, keepdim=True)
        patch -= patch_mean

        # calculate covariance matrix
        # cov = torch.cov(patch)
        # cov_eig_val, cov_eig_vec = torch.linalg.eig(cov)
        # # eigenvalues -> lambda = s**2 / n_v
        # lam = cov_eig_val / n_v
        # # calculate inequality, 3 batch dimensions!
        # left = (lam[:, :, :, 1:] - lam[:, :, :, -1, None]) / left_b[None, None, None]
        # right = right_a[None, None, None] * torch.cumsum(lam, dim=-1)[:, :, :, 1:]
        # # minimum p for which left < right
        # # we actually find all p for which left > right and set those 0 in s
        # # p = torch.argmax((left - right < 0).to(torch.int), dim=-1)
        # p = left < right
        # cov_eig_vec[:, :, :, :-1][p] = 0.0
        # cov_eig_vec[:, :, :, -1] = 0.0
        # # get coil compression matrix
        # d = torch.einsum("iklmn, om -> iklon", patch, cov_eig_vec)

        # do svd
        u, s, v = torch.linalg.svd(patch, full_matrices=False)
        # eigenvalues -> lambda = s**2 / n_v
        lam = s ** 2 / n_v
        if config.fixed_p > 0:
            # we use the p first singular values
            s[:, :, :, p:] = 0.0
            num_p = torch.full((nz, nch, ncx), p)
            theta_p = 1 / (1 + num_p)
        else:
            # calculate inequality, 3 batch dimensions!
            left = (lam[:, :, :, 1:] - lam[:, :, :, -1, None]) / left_b[None, None, None]
            r_lam = torch.einsum('is, czxs -> czxi', r_cumsum, lam)
            right = right_a[None, None, None] * r_lam
            # minimum p for which left < right
            # we actually find all p for which left < right and set those 0 in s
            p = left < right
            s[:, :, :, :-1][p] = 0.0
            s[:, :, :, -1] = 0.0
            num_p = torch.argmax(p.to(torch.int), dim=-1).cpu()
            theta_p = 1 / (1 + num_p.to(torch.float))
        # calculate denoised data, two batch dims!
        d = torch.matmul(torch.einsum("ijklm, ijkm -> ijklm", u, s.to(torch.complex128)), v)
        # add mean
        d += patch_mean
        # shape back
        d = torch.movedim(d, -2, -1)
        d = torch.reshape(d, patch_shape).cpu()

        # d = torch.movedim(d, 0, -2)
        # collect
        # dims [ch, z, c , c, m]
        # using the multipoint - pointwise approach of overlapping sliding window blocks from manjon et al. 2013
        # we summarize the contributions of each block at the relevant positions weighted by the
        # inverse number of nonzero coefficients in the diagonal eigenvalue / singular value matrix. i.e. P
        for idx_x in range(x_steps.shape[0]):
            data_denoised[:, :, x_steps[idx_x], idx_y:idx_y+cube_side_len] += (
                    theta_p[:, :, idx_x, None, None, None] * d[:, :, idx_x])
            data_access[:, :, x_steps[idx_x], idx_y:idx_y+cube_side_len] += theta_p[:, :, idx_x, None, None]
            data_p[:, :, x_steps[idx_x], idx_y:idx_y+cube_side_len] += num_p[:, :, idx_x, None, None]
            data_p_avg[:, :, x_steps[idx_x], idx_y:idx_y+cube_side_len] += 1

    data_denoised.real = torch.nan_to_num(
        torch.divide(data_denoised.real, data_access[:, :, :, :, None]),
        nan=0.0, posinf=0.0
    )
    data_denoised.imag = torch.nan_to_num(
        torch.divide(data_denoised.imag, data_access[:, :, :, :, None]),
        nan=0.0, posinf=0.0
    )

    data_denoised = torch.movedim(data_denoised, (0, 1), (2, 3))
    data_p_img = torch.nan_to_num(
        torch.divide(data_p, data_p_avg),
        nan=0.0, posinf=0.0
    )
    data_p_img = torch.movedim(data_p_img, (0, 1), (2, 3))

    # [x, y, z, ch, t]
    rsos = torch.sqrt(
        torch.sum(
            torch.square(
                torch.abs(
                    data_denoised
                )
            ),
            dim=-2
        )
    )
    # save rsos
    rsos *= 1000.0 / torch.max(rsos)
    img = nib.Nifti1Image(rsos.numpy(), affine=affine.numpy())
    file_path = save_path.joinpath(name).with_suffix(".nii")
    logging.info(f"write file: {file_path.as_posix()}")
    nib.save(img, filename=file_path.as_posix())
    # save access data
    img = nib.Nifti1Image(data_p_img.numpy(), affine=affine.numpy())
    file_path = save_path.joinpath(f"{name}_avg_p").with_suffix(".nii")
    logging.info(f"write file: {file_path.as_posix()}")
    nib.save(img, filename=file_path.as_posix())

    file_name = save_path.joinpath(name).with_suffix(".pt")
    logging.info(f"write file: {file_name.as_posix()}")
    torch.save(data_denoised, file_name.as_posix())


def batch_cov_removed_mean(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("___________________________________________________________________")
    logging.info("________________________ MP PCA Denoizing  ________________________")
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
