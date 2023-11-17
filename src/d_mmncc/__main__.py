from d_mmncc import options, algorithm
import logging
import pathlib as plib
import nibabel as nib
import numpy as np


def main(opts: options.Opts):
    logging.info(f"___MM Nc Chi Denoising___")
    logging.info(f"_________________________")
    # setup file path
    opts.display()
    input_file = plib.Path(opts.input_file).absolute()
    input_stem = input_file.stem
    output_path = plib.Path(opts.output_path).absolute()

    logging.info(f"load data: {input_file.as_posix()}")
    nii_img = nib.load(input_file.as_posix())
    nii_data = nii_img.get_fdata()
    nii_affine = nii_img.affine

    if np.max(nii_data) < 10:
        logging.info(f"rescaling data to avoid float / calc errors")
        rescale_factor = np.max(nii_data) / 1000
        nii_data /= rescale_factor
    else:
        rescale_factor = 1.0

    logging.info(f"setup algorithm")
    denoizer = algorithm.Denoizer(config=opts)

    logging.info(f"run")
    for num_iter in range(opts.num_max_runs):
        denoizer.get_nc_stats(data=nii_data, run=num_iter)
        if denoizer.check_low_noise(data_max=np.max(nii_data) / 8):
            break
        logging.info(f"denoize iteration: {num_iter + 1}")
        nii_data = denoizer.denoize_nii_data(
            data=nii_data
        )

    # scaling back
    nii_data *= rescale_factor
    logging.info(f"saving output")
    img = nib.Nifti1Image(nii_data, nii_affine)
    output_file = output_path.joinpath(f"d_{input_stem}").with_suffix(".nii")
    logging.info(f"writing file: {output_file.as_posix()}")
    nib.save(img, output_file.as_posix())
    logging.info(f"finished")


if __name__ == '__main__':
    parser, config = options.create_cli()

    if config.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=level)

    try:
        main(opts=config)
    except Exception as e:
        logging.error("Exception raised - revise usage:")
        parser.print_help()
        logging.exception(e)
        exit(e)
