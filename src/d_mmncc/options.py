""" options for usage as CLI"""
import simple_parsing as sp
import dataclasses as dc
import logging
import pandas as pd
import pathlib as plib
log_module = logging.getLogger(__name__)
pd.options.display.max_colwidth = 100


@dc.dataclass
class Opts(sp.helpers.Serializable):
    input_file: str = sp.field(default="examples/phant_acq-fa135_echo-01_MESE.nii", alias="-i", help=f"Input File Path")
    output_path: str = sp.field(default="", alias="-o", help=f"Output Directory")
    file_prefix: str = sp.field(default="d_", alias="-fp", help=f"Output file prefix appended to name after denoising")
    num_max_runs: int = sp.field(default=4, alias="-nmr", help=f"maximum number of denoising iterations")
    mp: bool = sp.field(default=True, help="usage of multiprocessing")
    mp_headroom: int = sp.field(default=4, alias="-mph", help="minimal unused cores when using multiprocessing")

    visualize: bool = sp.field(default=True, alias="-v", help="toggle show plots")
    debug: bool = sp.field(default=False, alias="-d", help="toggle logging debug information")
    single_iteration: bool = True

    def __post_init__(self):
        self._check_dirs()

    @classmethod
    def from_cli(cls, args: sp.ArgumentParser.parse_args):
        opts = args.options
        return opts

    def display(self):
        log_module.info(f"___ Configuration ___")
        df = pd.Series(self.to_dict())
        log_module.info(f"_____________________ \n{df}")

    def _check_dirs(self):
        in_path = plib.Path(self.input_file).absolute()
        if not in_path.is_file():
            err = f"input file {in_path.as_posix()} not found"
            logging.error(err)
            raise FileNotFoundError(err)
        else:
            self.input_file = in_path.as_posix().__str__()

        # no output path provided
        if not self.output_path:
            log_module.info(f"no output path provided, using same as input")
            self.output_path = in_path.parent.as_posix().__str__()
        else:
            output_path = plib.Path(self.output_path).absolute()
            if output_path.suffix:
                # if filename not path given, use only path
                output_path = output_path.parent
            # make if not exist
            output_path.mkdir(parents=True, exist_ok=True)


def create_cli() -> (sp.ArgumentParser, Opts):
    parser = sp.ArgumentParser(prog="denoiser -- non-central-chi-majorize-minimize")
    parser.add_arguments(Opts, dest="options")
    args = parser.parse_args()
    opts = args.options

    return parser, opts


if __name__ == '__main__':
    p, o = create_cli()
    if o.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=level)
    p.print_help()
    o.display()

    def_path = plib.Path("./examples/config.json").absolute()
    o.save_json(def_path.as_posix(), indent=2)
