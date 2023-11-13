from __future__ import annotations

import gc
import os
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from skyfield.api import load, wgs84, utc

plot_kwargs = {
    "dt": "Datetime object representing a time of an observation. If None - will not be specified under plot.",
    "pos": "List containing geographical latitude [deg], longitude[deg] and altitude[m] representing a position of "
           " an instrument. If None - will not be specified under plot.",
    "freq": "Float representing a frequency of an observation. If None - will not be specified under plot.",
    "title": "Title of the plot",
    "barlabel": "Label near colorbar. Most functions override this parameter.",
    "plotlabel": "Label on the plot. Usually includes date/time, location and frequency of an observation. If None - "
                 "will not appear.",
    "cblim": "Tuple containing min and max values of the colorbar scale. If any of limits is None - max/min values is "
             "automatically calculated.",
    "saveto": "Path to save the plot. Must include name. If not specified - the plot will not be saved.",
    "dpi": "Image resolution.",
    "cmap": "A colormap to use in the plot.",
    "cbformat": "Formatter of numbers on the colorbar scale.",
    "nancolor": "A color to fill np.nan in the plot.",
    "infcolor": "A color to fill np.inf in the plot.",
    "local_time": "Difference between local time and UTC. If specified - local time is shown instead of UTC.",
    "cinfo": "If true - places model info in the centre of the picture.",
    "lfont": "If true - the font size of labels is increased.",
    "cbar": "If true - a colorbar is added.",
    "sunpos": "If True - the position of the sun is plotted. Dashed line if the Sun is below horizon.",
}


def polar_plot(
        data: Sequence[np.ndarray, np.ndarray, np.ndarray],
        dt: datetime | None = None,
        pos: Sequence[float, float, float] | None = None,
        freq: float | None = None,
        title: str | None = None,
        barlabel: str | None = None,
        plotlabel: str | None = "",
        cblim: Sequence[float, float] = None,
        saveto: str | None = None,
        dpi: int = 300,
        cmap: str = "plasma",
        cbformat: str = None,
        nancolor: str = "black",
        infcolor: str = "white",
        local_time: int | None = None,
        cinfo: bool = False,
        lfont: bool = False,
        cbar: bool = True,
        sunpos: bool = False,
):
    """
    A core function for graphic generation on the visible sky field.
    See all available option listed in dionpy.plot_kwargs.

    :return: A matplotlib figure.
    """
    cblim = cblim or [None, None]
    if cblim[0] is None:
        cblim[0] = np.nanmin(data[2][data[2] != -np.inf])
    if cblim[1] is None:
        cblim[1] = cblim[1] or np.nanmax(data[2][data[2] != np.inf])

    plot_data = np.where(np.isinf(data[2]), cblim[1] + 1e8, data[2])
    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    cmap.set_bad(nancolor)
    cmap.set_over(infcolor)

    # Finding square sum color in the center
    datarel = data[2][0, -1] / data[2].max()
    cl = colormaps.get_cmap(cmap)(datarel)
    cl_sum = np.sum(np.array(cl) ** 2) / 4
    labelcolor = 'black' if cl_sum > 0.7 else 'white'

    masked_data = np.ma.array(plot_data, mask=np.isnan(plot_data))

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
    ax.set_theta_zero_location("N")
    rfmt = lambda x_, _: f"{x_:.0f}°"
    ax.yaxis.set_major_formatter(FuncFormatter(rfmt))
    ax.set_theta_direction(-1)

    labelsize = 14 if lfont else 11
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    ax.tick_params(axis="y", which="major", labelcolor=labelcolor)
    if cbar:
        cbar = plt.colorbar(img, fraction=0.042, pad=0.08, format=cbformat)
    cbar.set_label(label=barlabel, size=labelsize)
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.ax.yaxis.get_offset_text().set_fontsize(labelsize)

    title_pad = 20 if cinfo else 55
    plt.title(title, fontsize=14, pad=title_pad)

    if plotlabel == "":
        if pos is not None:
            plotlabel += f"Lat/Lon: {pos[0]:.3f}, {pos[1]:.3f}"
        if dt is not None:
            if local_time is None:
                plotlabel += (
                        "\nUTC time: " + datetime.strftime(dt, "%Y-%m-%d %H:%M")
                )
            elif isinstance(local_time, int):
                plotlabel += (
                        "\nLocal t: "
                        + datetime.strftime(
                    dt + timedelta(hours=local_time), "%Y-%m-%d %H:%M"
                )
                )
        if freq is not None:
            plotlabel += f"\nFrequency: {freq:.1f} MHz"

    if plotlabel is not None:
        if cinfo:
            ax.set_rticks([20, 40, 60, 80])
            ax.grid(linestyle=":", alpha=0)
            props = dict(boxstyle='round', facecolor='black', alpha=0.0)
            textprorps = dict(transform=ax.transAxes, ha="center", va="center", bbox=props, family='monospace',
                              fontsize=11, color=labelcolor)
            plt.text(0.5, 0.5, plotlabel, **textprorps)
            _custom_grid(ax, color=labelcolor)
        else:
            ax.set_xticks(np.arange(0, 8) * np.pi / 4)
            ax.set_xticklabels([""] + [str(x) + r"$^\circ$" for x in np.arange(1, 8) * 45])
            ax.set_rticks([0, 30, 60, 90])
            ax.grid(linestyle=":", alpha=0.7, color=labelcolor)
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            textprorps = dict(transform=ax.transAxes, bbox=props, ha="center", va="center",
                              family='monospace', color='black', fontsize=labelsize)
            infoxpos = 0.5
            infoypos = 1.06
            plt.text(infoxpos, infoypos, plotlabel, **textprorps)

    else:
        ax.set_rticks([0, 30, 60, 90])
        ax.grid(linestyle=":", alpha=0.7, color=labelcolor)

    if sunpos:
        ts = load.timescale()
        sf_dt = dt.replace(tzinfo=utc)
        sf_time = ts.from_datetime(sf_dt)
        sf_pos = wgs84.latlon(pos[0], pos[1])
        sf_planets = load('de421.bsp')
        earth, sun = sf_planets['earth'], sf_planets['sun']
        obs_loc = earth + sf_pos
        sunalt, sunaz, _ = obs_loc.at(sf_time).observe(sun).apparent().altaz()
        sunalt, sunaz = sunalt.degrees, sunaz.degrees

        if sunalt < 0:
            sunalt = np.abs(sunalt)
            linestyle = ':'
        else:
            linestyle = '-'
        plt.scatter(np.deg2rad(sunaz), 90 - sunalt, s=20, facecolors=labelcolor, lw=0)
        plt.scatter(np.deg2rad(sunaz), 90 - sunalt, s=180, facecolors='none', edgecolors=labelcolor, lw=2,
                    linestyle=linestyle)

    if saveto is not None:
        head, tail = os.path.split(saveto)
        if not os.path.exists(head) and len(head) > 0:
            os.makedirs(head)
        plt.savefig(saveto, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        gc.collect()
        return
    return fig


def _custom_grid(ax: plt.Axes, color='gray'):
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
    r = np.ones(theta.shape)
    gridprop = dict(alpha=0.7, color=color, linestyle=':', linewidth=0.75)
    gtdelta = np.pi / 4
    ax.plot(theta, np.where(
        ((theta < np.pi / 2 + gtdelta) & (theta > np.pi / 2 - gtdelta)) |
        ((theta < 3 * np.pi / 2 + gtdelta) & (theta > 3 * np.pi / 2 - gtdelta)),
        np.nan, r * 20), **gridprop)
    ax.plot(theta, r * 40, **gridprop)
    ax.plot(theta, r * 60, **gridprop)
    ax.plot(theta, r * 80, **gridprop)

    r = np.linspace(20, 90, 50, endpoint=True)
    r2 = np.linspace(40, 90, 50, endpoint=True)
    ax.plot(0 * r + np.pi / 4 * 0, r, **gridprop)
    ax.plot(0 * r + np.pi / 4 * 1, r, **gridprop)
    ax.plot(0 * r + np.pi / 4 * 2, r2, **gridprop)
    ax.plot(0 * r + np.pi / 4 * 3, r, **gridprop)
    ax.plot(0 * r + np.pi / 4 * 4, r, **gridprop)
    ax.plot(0 * r + np.pi / 4 * 5, r, **gridprop)
    ax.plot(0 * r + np.pi / 4 * 6, r2, **gridprop)
    ax.plot(0 * r + np.pi / 4 * 7, r, **gridprop)


def polar_plot_star(args):
    return polar_plot(args[:3], *args[3:-1], **args[-1])
