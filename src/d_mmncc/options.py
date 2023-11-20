import simple_parsing as sp
import pandas as pd
import logging
import pathlib as plib
import dataclasses as dc
log_module = logging.getLogger(__name__)


@dc.dataclass
class Config(sp.Serializable):
    """
        Configuration for denoising
        """
    config_file: str = sp.field(
        alias="-c", default="",
        help="provide Configuration file (.json)"
    )
    nii_path: str = sp.field(
        alias="-i", default="",
        help="set filepath to nii file"
    )
    save_path: str = sp.field(
        alias="-s", default="",
        help="set path to save files (optional, if blank use input path)"
    )
    file_prefix: str = sp.field(
        default="d", alias="-fp",
        help=f"Output file prefix appended to name after denoising"
    )

    method: int = sp.field(
        default="cp", alias="-m", choices=["cp"],
        help=f"solver for l2 minimization"
    )
    solver_max_num_iter: int = sp.field(
        alias="-smn", default=50, help="solver algorithm (via method): - maximum number of iterations within"
                                       "(inner loop)"
    )
    solver_tv_lambda: float = sp.field(
        alias="-stv", default=0.01, help='solver algorithm (via method): - weight of total variation within '
                                         '(inner loop)'
    )
    max_num_iter: int = sp.field(
        alias="-mn", default=10, help="iterative minimization procedure, each step solved via solver algorithm,"
                                      "max number of steps."
    )
    mp: bool = sp.field(
        default=True, help="usage of multiprocessing"
    )
    mp_headroom: int = sp.field(
        default=8, alias="-mph",
        help="minimal unused cores when using multiprocessing"
    )

    visualize: bool = sp.field(
        default=True, alias="-v", help="toggle show plots"
    )
    debug: bool = sp.field(
        default=False, alias="-d", help="toggle logging debug information"
    )

    @classmethod
    def from_cli(cls, args: sp.ArgumentParser.parse_args):
        instance = args.config
        if instance.config_file:
            c_path = plib.Path(instance.config_file).absolute()
            if not c_path.is_file():
                err = f"Config File set: {c_path.as_posix()} not found!"
                log_module.error(err)
                raise FileNotFoundError(err)
            instance = cls.load(c_path.as_posix())
        if not instance.save_path:
            instance.save_path = plib.Path(instance.nii_path).absolute().parent.as_posix()
        instance.display()
        return instance

    def display(self):
        # display via logging
        df = pd.Series(self.to_dict())
        # concat empty entry to start of series for nicer visualization
        df = pd.concat([pd.Series([""], index=["___ Config ___"]), df])
        # display
        log_module.info(df)


def create_cli() -> (sp.ArgumentParser, sp.ArgumentParser.parse_args):
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    parser = sp.ArgumentParser(prog='denoising_mm_autodmri')
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()
    return parser, args

