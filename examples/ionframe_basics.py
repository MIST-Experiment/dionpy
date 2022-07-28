import matplotlib.pyplot as plt
import numpy as np

from onion import IonFrame
from datetime import datetime

# Specifying date and time of observation:
dt = datetime(year=2018, month=2, day=13, hour=6, minute=20)
# Another possible way:
# dt = datetime.strptime("2018-02-13 06:20", "%Y-%m-%d %H:%M")
# For more details see datetime documentation.

# Defining instrument position - lat [deg], lon [deg], alt [m]:
pos = (79.5, 0, 0)

# Frequency of observation in [Hz]
freq = 100e6

# Defining an IonFrame model - will automatically perform parallel calculation
model = IonFrame(dt, pos)

el, az = np.meshgrid(np.linspace(0, 90, 100), np.linspace(0, 360, 100))
att = model.atten(el, az, freq)
plt.pcolormesh(el, az, att)
plt.colorbar()
plt.show()

model.plot_atten(freq)
plt.show()

