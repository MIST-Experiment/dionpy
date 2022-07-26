import os
from datetime import datetime
from typing import Tuple, Union

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt

plot_kwargs = {
    "dt": "Datetime object representing a time of an observation. If None - will not be specified under plot.",
    "pos": "List containing geographical latitude [deg], longitude[deg] and altitude[m] representing a position of "
    " an instrument. If None - will not be specified under plot.",
    "freq": "Float representing a frequency of an observation. If None - will not be specified under plot.",
    "title": "Title of the plot",
    "barlabel": "Label near colorbar. Most functions override this parameter.",
    "plotlabel": "Label under plot. Usually includes date/time, location and frequency of an observation.",
    "cblim": "Tuple containing min and max values of the colorbar scale.",
    "saveto": "Path to save the plot. Must include name. If not specified - the plot will not be saved.",
    "dpi": "Image resolution.",
    "cmap": "A colormap to use in the plot.",
    "cbformat": "Formatter of numbers on the colorbar scale.",
    "nancolor": "A color to fill np.nan in the plot.",
}


def polar_plot(
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    dt: Union[datetime, None] = None,
    pos: Union[Tuple[float, float, float], None] = None,
    freq: Union[float, None] = None,
    title: Union[str, None] = None,
    barlabel: Union[str, None] = None,
    plotlabel: Union[str, None] = None,
    cblim: Union[Tuple[float, float], None] = None,
    saveto: Union[str, None] = None,
    dpi: int = 300,
    cmap: str = "viridis",
    cbformat: str = None,
    nancolor: str = "black",
):
    """
    A core function for graphic generation on the visible sky field.
    See all available option listed in mistion.plot_kwargs.

    :return: A matplotlib figure.
    """
    if plotlabel is None:
        plotlabel = ""
        if pos is not None:
            plotlabel += f"Position: lat={pos[0]:.3f}, lon={pos[1]:.3f}\n"
        if dt is not None:
            plotlabel += "UTC time: " + datetime.strftime(dt, "%Y-%m-%d %H:%M") + "\n"
        if freq is not None:
            plotlabel += f"Frequency: {freq / 1e6:.1f} MHz"

    cblim = cblim or (np.nanmin(data[2]), np.nanmax(data[2]))

    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    cmap.set_bad(nancolor, 1.0)

    masked_data = np.ma.array(data[2], mask=np.isnan(data[2]))

    fig = plt.figure(figsize=(8, 8))
    ax: plt.Axes = fig.add_subplot(111, projection="polar")
    img = ax.pcolormesh(
        data[0],
        data[1],
        masked_data,
        cmap=cmap,
        vmin=cblim[0],
        vmax=cblim[1],
        shading="auto",
    )
    ax.grid(color="gray", linestyle=":")
    ax.set_theta_zero_location("S")
    ax.set_rticks([90, 60, 30, 0], Fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="y", which="major", labelcolor="gray")
    plt.colorbar(img, fraction=0.042, pad=0.08, format=cbformat).set_label(
        label=barlabel, size=10
    )
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(plotlabel, fontsize=10)
    if saveto is not None:
        head, tail = os.path.split(saveto)
        if not os.path.exists(head):
            os.makedirs(head)
        plt.savefig(saveto, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return
    return fig


def polar_plot_star(pars):
    return polar_plot(*pars)
