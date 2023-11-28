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
import plotly.express as px
import numpy as np
import tqdm
from d_mppca import options
import torch
import nibabel as nib
import itertools

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
    path = plib.Path("/data/pt_np-jschmidt/data/01_in_vivo_scan_data/"
                     "pulseq_mese_megesse/7T/pulseq_2023-11-13/raw/megesse_fs/").absolute()
    file_path = path.joinpath("k_space.pt")
    logging.info(f"load file: {file_path}")
    img_data = torch.load(file_path.as_posix())

    file_path = path.joinpath("affine.pt")
    logging.info(f"load file: {file_path}")
    affine = torch.load(file_path.as_posix())

    if config.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logging.info("fft to image space")
    img_shape = img_data.shape
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

    # we need to batch the data to fit on memory, easiest is to do it slice based
    # want to batch channel dim and two of the dimensional axis
    # get vars
    m = img_shape[-1]
    cube_side_len = torch.ceil(torch.sqrt(torch.tensor([m]))).to(torch.int).item()
    n_v = cube_side_len ** 2
    # sliding window, want to move the patch through each dim
    cube_steps_x = torch.arange(0, img_shape[0] - cube_side_len)
    cube_steps_x = cube_steps_x[:, None] + torch.arange(cube_side_len)[None, :]
    cube_steps_y = torch.arange(0, img_shape[1] - cube_side_len)
    cube_steps_y = cube_steps_y[:, None] + torch.arange(cube_side_len)[None, :]

    # sliding window combinations
    sliding_window = list(itertools.product(cube_steps_x, cube_steps_y))
    # calculate const for mp inequality
    left_b = 4 * torch.sqrt(torch.tensor((m - torch.arange(m - 1)) / n_v)).to(device)
    right_a = (1 / (m - torch.arange(m - 1))).to(device)
    # batch dim channels, slice, sliding window combinations
    img_data = torch.movedim(img_data, (3, 2), (0, 1))
    img_data = torch.reshape(img_data, (-1, *img_shape[:2], m))
    data_access = torch.zeros_like(img_data, dtype=torch.float)
    data_denoised = torch.zeros_like(img_data)
    # batch
    batch_size = 2000
    b_sw = torch.tensor_split(torch.tensor(sliding_window), batch_size)
    # dims [b, ch, z, cx, cy, m]
    for combs in tqdm.trange(batch_size, desc='batch processing patches', position=0, leave=False):
        end_xy = combs + cube_side_len

        # patches = torch.zeros_like(batch_data)
        # for idx_comb in range(combs.shape[0]):
        #     start_x, start_y = combs[idx_comb]
        #     end_x = start_x + cube_side_len
        #     end_y = start_y + cube_side_len
        #     patches[idx_comb] = img_data[:, :, start_x:end_x, start_y:end_y]
        # try batched svd
        # patch = img_data[:, :, start_x:end_x, start_y:end_y].to(device)
        patch_shape = patch.shape
        # dims [ch, z, patch-size, m]
        patch = torch.reshape(patch, (*patch_shape[:2], -1, m))
        # remove mean across spatial dim of patch
        patch_mean = torch.mean(patch, dim=-2, keepdim=True)
        patch -= patch_mean
        # do svd
        u, s, v = torch.linalg.svd(patch, full_matrices=False)
        # eigenvalues -> lambda = s**2 / n_v
        lam = s ** 2 / n_v
        # calculate inequality, 2 batch dimensions!
        left = (lam[:, :, 1:] - lam[:, :, -1, None]) / left_b[None, None]
        right = right_a[None, None] * torch.cumsum(lam, dim=-1)[:, :, 1:]
        # minimum p for which left < right
        # we actually find all p for which left > right and set those 0 in s
        # p = torch.argmax((left - right < 0).to(torch.int), dim=-1)
        p = left < right
        s[:, :, :-1][p] = 0.0
        s[:, :, -1] = 0.0
        # calculate denoised data, two batch dims!
        d = torch.matmul(torch.einsum("iklm, ikm -> iklm", u, s.to(torch.complex128)), v)
        # add mean
        d += patch_mean
        # shape back
        num_p = torch.argmax(p.to(torch.int), dim=-1)
        num_p = 1 / (1 + num_p.to(torch.float))
        # d = torch.movedim(d, 0, -2)
        d = torch.reshape(d, patch_shape)
        # collect
        # dims [ch, z, c , c, m]
        # using the multipoint - pointwise approach of overlapping sliding window blocks from manjon et al. 2013
        # we summarize the contributions of each block at the relevant positions weighted by the
        # inverse number of nonzero coefficients in the diagonal eigenvalue / singular value matrix. i.e. P
        data_access[:, :, start_x:end_x, start_y:end_y] += num_p[:, :, None, None, None]
        data_denoised[:, :, start_x:end_x, start_y:end_y] += num_p[:, :, None, None, None] * d

    data_denoised.real = torch.nan_to_num(data_denoised.real / data_access, nan=0.0, posinf=0.0)
    data_denoised.imag = torch.nan_to_num(data_denoised.imag / data_access, nan=0.0, posinf=0.0)

    data_denoised = torch.movedim(data_denoised, (0, 1), (3, 2))

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
    file_path = path.joinpath("denoised_mppca_rsos").with_suffix(".nii")
    logging.info(f"write file: {file_path.as_posix()}")
    nib.save(img, filename=file_path.as_posix())

    file_name = path.joinpath("denoised_mppca").with_suffix(".pt")
    logging.info(f"write file: {file_name.as_posix()}")
    torch.save(data_denoised, file_name.as_posix())


def main_old():
    for slice_idx in tqdm.trange(k_shape[2], desc="slice processing", leave=False, position=0):
        # want channel dim & y as batch dimensions . loop over z and x
        img_slice = img_data[:, :, slice_idx].to(device)

        data_slice = torch.zeros_like(img_slice, device=device)
        data_access = torch.zeros_like(img_slice, dtype=torch.float, device=device)
        # patch based. we extract squares of the next highest squared order compared to the combined echo channel size

        # dims [ch, y batch, patchx, patchy, cube, m]
        for cube_step_x in tqdm.tqdm(cube_steps_x, desc='batch processing patches', position=1, leave=False):
            end_x = cube_step_x + cube_side_len
            # try batched svd
            patches = torch.zeros((cube_steps_y.shape[0], cube_side_len, cube_side_len, k_shape[-2], m),
                                  device=device, dtype=img_slice.dtype)
            for idy in range(cube_steps_y.shape[0]):
                end_y = cube_steps_y[idy] + cube_side_len
                patches[idy] = img_slice[cube_step_x:end_x, cube_steps_y[idy]:end_y]
            patch_shape = patches.shape
            # move channels to front
            patches = torch.movedim(patches, -2, 0)
            # dims [ch, y, patch-size, m]
            patches = torch.reshape(patches, (patches.shape[0], cube_steps_y.shape[0], -1, m))
            # remove mean across spatial dim of patch
            patch_mean = torch.mean(patches, dim=-2, keepdim=True)
            patches = patches - patch_mean
            # do svd
            u, s, v = torch.linalg.svd(patches, full_matrices=False)
            # eigenvalues -> lambda = s**2 / n_v
            lam = s**2 / n_v
            # calculate inequality, 2 batch dimensions!
            left = (lam[:, :, 1:] - lam[:, :, -1, None]) / left_b[None, None]
            right = right_a[None, None] * torch.cumsum(lam, dim=-1)[:, :, 1:]
            # minimum p for which left < right
            # we actually find all p for which left > right and set those 0 in s
            # p = torch.argmax((left - right < 0).to(torch.int), dim=-1)
            p = left < right
            s[:, :, :-1][p] = 0.0
            s[:, :, -1] = 0.0
            # calculate denoised data, two batch dims!
            d = torch.matmul(torch.einsum("iklm, ikm -> iklm", u, s.to(torch.complex128)), v)
            # add mean
            d += patch_mean
            # shape back
            num_p = torch.argmax(p.to(torch.int), dim=-1)
            num_p = 1 / (1 + torch.movedim(num_p, 0, -1).to(torch.float))
            d = torch.movedim(d, 0, -2)
            d = torch.reshape(d, patch_shape)
            # collect
            # dims [y batch, c, c , m]
            for idy in range(cube_steps_y.shape[0]):
                end_y = cube_steps_y[idy] + cube_side_len
                # using the multipoint - pointwise approach of overlapping sliding window blocks from manjon et al. 2013
                # we summarize the contributions of each block at the relevant positions weighted by the
                # inverse number of nonzero coefficients in the diagonal eigenvalue / singular value matrix. i.e. P
                data_access[cube_step_x:end_x, cube_steps_y[idy]:end_y] += num_p[idy][None, None, :, None]
                data_slice[cube_step_x:end_x, cube_steps_y[idy]:end_y] += num_p[idy][None, None, :, None] * d[idy]
        tmp = torch.divide(
            data_slice,
            data_access
        )
        dd = torch.reshape(tmp.cpu(), (*k_shape[:2], *k_shape[-2:]))
        denoised_data[:, :, slice_idx].real = torch.nan_to_num(dd.real, nan=0.0, posinf=0.0)
        denoised_data[:, :, slice_idx].imag = torch.nan_to_num(dd.imag, nan=0.0, posinf=0.0)

        if slice_idx == 0:
            fig = px.imshow(np.abs(denoised_data[:, :, slice_idx, :8, 0].numpy(force=True)), facet_col=2,
                            facet_col_wrap=4)

            file_name = path.joinpath("denoised_mppca_slice_0").with_suffix(".html")
            logging.info(f"write file: {file_name.as_posix()}")
            fig.write_html(file_name.as_posix())
            r_slice_fft = torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(
                        img_data[:, :, slice_idx, :8, 0]
                    ),
                    dim=(0, 1)
                )
            )
            fig = px.imshow(np.abs(r_slice_fft.numpy(force=True)), facet_col=2, facet_col_wrap=4)
            file_name = path.joinpath("reference_slice_0").with_suffix(".html")
            logging.info(f"write file: {file_name.as_posix()}")
            fig.write_html(file_name.as_posix())

    # [x, y, z, ch, t]
    rsos = torch.sqrt(
        torch.sum(
            torch.square(
                torch.abs(
                    denoised_data
                )
            ),
            dim=-2
        )
    )
    # save rsos
    rsos *= 1000.0 / torch.max(rsos)
    img = nib.Nifti1Image(rsos.numpy(), affine=affine.numpy())
    file_path = path.joinpath("denoised_mppca_rsos").with_suffix(".nii")
    logging.info(f"write file: {file_path.as_posix()}")
    nib.save(img, filename=file_path.as_posix())

    file_name = path.joinpath("denoised_mppca").with_suffix(".pt")
    logging.info(f"write file: {file_name.as_posix()}")
    torch.save(denoised_data, file_name.as_posix())
    # img = nib.Nifti1Image(denoise_data, affine)
    # nib.save(img, file_name.as_posix())


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
