import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psub
from .options import Config
import pathlib as plib
import logging

log_module = logging.getLogger(__name__)


def set_fig_path(config: Config) -> plib.Path:
    out_path = plib.Path(config.save_path)
    fig_path = out_path.joinpath("plots")
    fig_path.mkdir(exist_ok=True, parents=True)
    return fig_path


def check_name(name: str):
    if not name:
        err = "no filename given to save plot to"
        log_module.error(err)
        raise AttributeError(err)


def plot_img(data: np.ndarray, config: Config, name: str):
    """
    data assumed to be 2d, 3d or 4d.
    if 2d we plot an img
    if 3d we plot facet columns with third dimension
    if 4d we plot grid with 3rd d in rows and 4th d in columns
    """
    check_name(name=name)
    file_name = set_fig_path(config=config).joinpath(name).with_suffix(".html")

    if len(data.shape) == 2:
        fig = px.imshow(data)
    elif len(data.shape) == 3:
        fig = px.imshow(data, facet_col=2, facet_col_wrap=10)
    elif len(data.shape) == 4:
        fig = psub.make_subplots(rows=data.shape[2], cols=data.shape[3])
        for r_idx in range(data.shape[2]):
            for c_idx in range(data.shape[3]):
                fig.add_trace(
                    go.Heatmapgl(z=data[:, :, r_idx, c_idx], name=f"{r_idx}: {c_idx}")
                )
                fig.update_xaxes(visible=False).update_yaxes(visible=False)

    else:
        err = "data dimension not suited for plotting img, need 2 - 4 dimensional data"
        log_module.error(err)
        raise AttributeError(err)

    log_module.info(f"Writing file: {file_name.as_posix()}")
    fig.write_html(file_name.as_posix())


def plot_noise_sigma_n(sigma: np.ndarray, n: np.ndarray,
                       config: Config, name: str):
    """
    both inputs supposed to be 1D
    """
    check_name(name=name)
    file_name = set_fig_path(config=config).joinpath(name).with_suffix(".html")

    if len(sigma.shape) > 1 or len(n.shape) > 1:
        err = f"assumed 1D input"
        log_module.error(err)
        raise AttributeError(err)

    fig = psub.make_subplots(2, 1, shared_xaxes=True)
    fig.add_trace(
        go.Bar(x=np.arange(sigma.shape[0]), y=sigma, name="sigma"),
        1, 1
    )
    fig.add_trace(
        go.Bar(x=np.arange(n.shape[0]), y=n, name="N"),
        2, 1
    )

    # Add figure title
    fig.update_layout(
        title_text="Autodmri sigma & n"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="slice index")

    fig.update_yaxes(range=[0, 1.2 * np.max(sigma)], row=1, col=1, title_text="sigma")
    fig.update_yaxes(range=[0, 1.2 * np.max(n)], row=2, col=1, title_text="n")

    log_module.info(f"Writing file: {file_name.as_posix()}")
    fig.write_html(file_name.as_posix())


def plot_hist_fit(data: np.ndarray, config: Config, name: str,
                  fit_line: np.ndarray = None):
    check_name(name=name)
    file_name = set_fig_path(config=config).joinpath(name).with_suffix(".html")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=data, histnorm="probability")
    )

    if fit_line is not None:
        fig.add_trace(
            go.Scattergl(y=fit_line, name=f"fit")
        )

    log_module.info(f"write file: {file_name.as_posix()}")
    fig.write_html(file_name.as_posix())
