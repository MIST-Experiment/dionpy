Quickstart
============

.. code-block::

    from dionpy import IonFrame
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Date of observation
    dt = datetime(year=2022, month=7, day=17, hour=12, minute=0)

    # Instrument position: latitude [deg], longitude [deg], altitude [m]
    pos = (79.418, -90.810, 0)

    # Define a model
    model = IonFrame(dt, pos)

    # Plot ionospheric attenuation
    model.plot_atten(freq=40, title=r"Attenuation factor $f_a$")
    plt.show()

    # Plot ionospheric refraction
    model.plot_refr(freq=40, title=r"Refraction angle $\delta \theta$")
    plt.show()

