import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from d_mmncc import distributions
import logging
import typing
import pathlib as plib
log_module = logging.getLogger(__name__)


def plot_noise_histogram(noise_array: typing.Union[np.ndarray, list], path: typing.Union[str, plib.Path],
                         name: str = "", num_channels: int = 16, sigma: float = -1.0):
    """
    Plotting function to visualize flat array or list collection of noise points and their characteristics

    :param noise_array: numpy array or list of noise points
    :param path:
    :param num_channels:
    :param sigma:
    :return:
    """
    noise_array = np.asarray(noise_array)
    noise_array = noise_array.flatten()

    # build fit
    nc_chi = distributions.NcChi()
    nc_chi.set_channels(num_channels=num_channels)
    if sigma > 1e-5:
        # sigma given
        nc_chi.set_sigma(sigma=sigma)
    else:
        # per default assume sigma of half max noise
        nc_chi.set_sigma(sigma=np.max(noise_array) / 2)
    x_ax = np.linspace(0, np.max(noise_array * 1.2), 200)
    # build pdf
    chi_fit = nc_chi.pdf(x_ax, 0)

    fig1 = px.histogram(noise_array, histnorm="probability density", color_discrete_sequence=["#06d485"])
    fig2 = px.line(
        x=x_ax, y=chi_fit, color_discrete_sequence=["#6306d4"])
    fig = go.Figure(fig1.data + fig2.data)
    fig.update_layout(
        title="Noise Characteristics",
        xaxis_title="Voxel Value",
        yaxis_title="Count [%]",
        legend_title="Legend",
    )

    fig['data'][1]['showlegend'] = True
    fig['data'][1]['name'] = 'distribution fit'
    fig['data'][0]['name'] = 'data histogram'

    save_fig = plib.Path(path).absolute()
    save_fig.parent.mkdir(parents=True, exist_ok=True)
    if name:
        name = "_"+name
    save_fig = save_fig.joinpath(f"noise_histogram{name}").with_suffix(".html")

    log_module.info(f"Writing file: {save_fig.as_posix()}")
    fig.write_html(save_fig.as_posix())
