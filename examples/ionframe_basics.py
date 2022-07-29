import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from onion import IonFrame

# Before using the model one must specify set of basic parameters.

# 1) Date and time of observation:
dt = datetime(year=2018, month=2, day=13, hour=6, minute=20)
# Another possible way:
# dt = datetime.strptime("2018-02-13 06:20", "%Y-%m-%d %H:%M")
# For more details see datetime documentation.

# 2) Instrument position - lat [deg], lon [deg], alt [m]:
pos = (79.5, 0, 0)

# Defining an IonFrame model is simple! It will automatically perform calculations
# in parallel in the background
model = IonFrame(dt, pos)

# Let's get info about ionospheric attenuation at specified time. You can access it using
# IonFrame.atten() method. You mast pass elevation, azimuth and frequency as required
# parameters. The calculation will be more effective if elevation and azimuth are numpy
# arrays. Frequency can also be passed as an array - in that case the calculation will
# be performed in parallel.

# Frequency of observation in [Hz]
freq = 100e6
# Grid of elevations and azimuths
elevation, azimuth = np.meshgrid(np.linspace(0, 90, 100), np.linspace(0, 360, 100))
# Using all them to get an attenuation factor
attenuation = model.atten(elevation, azimuth, freq)

# Now we can do a simple visualization to see the result.
plt.pcolormesh(elevation, azimuth, attenuation)
plt.title("Simple data representation from scratch")
plt.colorbar()
plt.show()

# Doesn't look good. The IonFrame class implements several high-level methods to
# help you visualize data. See more examples in the plotting tutorial, but for now
# let's plot the attenuation.
model.plot_atten(freq, title="Fancy-looking pre-implemented plotter")
plt.show()

# The refraction of the ionosphere can be obtained in the same simple way -
# using IonFrame.refr() method. Want to take a look?
model.plot_refr(freq, title="Ionospheric refraction")
plt.show()

# Finally, if you don't want to recalculate the model every time, it is possible
# to save it to a file in a single line of code.
model.save(savedir="ion_models/", name="my_model")

# Loading models is also simple.
loaded_model = IonFrame.load("ion_models/my_model")
