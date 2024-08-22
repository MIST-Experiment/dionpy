More examples
=============

Plasma frequency
----------------
Start by defining an :class:`~dionpy.IonFrame` (see details in :ref:`guide-ionframe`).

.. code-block:: python

    from datetime import datetime
    from dionpy import IonFrame
    import matplotlib.pyplot as plt
    import numpy as np

    # UTC time
    dt = datetime(year=2024, month=7, day=13, hour=12)
    # Lat / lon / height above sea level
    pos = (79.456944, -90.800833, 0)

    # Currently plasfreq methods do not accept height as an input
    # (no vetical interpolation implemented), so you need to specify
    # the number of the pre-calculated sublayer.
    # For example, let's probe 10 layers between 100 and 1000 km.
    frame = IonFrame(dt, pos, hbot=100, htop=1000, nlayers=10)
    print(frame)    # Displays details of the IonFrame instance

Now let's try a simple plot function. It displays plasma frequency on
the line of sight at constant height above sea level (specified by layer number).

.. code-block:: python


    # Plotting plasma frequency
    fig1 = frame.plot_plasfreq(layer=0)
    fig2 = frame.plot_plasfreq(layer=4)
    fig3 = frame.plot_plasfreq(layer=9)
    plt.show()

You can also numerically calculate plasma frequency in given layer for
custom coordinate grid.

.. code-block:: python


    # Calculating plasma frequency
    # Define working coordinate grid
    alt = np.linspace(0, 90, 100)  # Altitude axis
    az = np.linspace(0, 360, 100)  # Azimuth axis
    alt_m, az_m = np.meshgrid(alt, az)  # Rectangular coordinate grid

    # Calculates angular (by default) plasma frequency in [Hz]
    pf = frame.plasfreq(alt_m, az_m, layer=4)

    # Displaying in colormesh for example
    plt.pcolormesh(alt_m, az_m, pf)
    plt.show()