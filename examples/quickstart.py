from dionpy import IonFrame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Date of observation
dt = datetime(year=2022, month=7, day=17, hour=12, minute=0)

# Instrument position: latitude [deg], longitude [deg], altitude [m]
pos = (79.418, -90.810, 0)

# Define a model
model = IonFrame(dt, pos)

# Define frequency of observation in [MHz]
freq = 40

# Plot ionospheric attenuation
model.plot_atten(freq, title=r"Attenuation factor $f_a$")
plt.show()

# Plot ionospheric refraction
model.plot_refr(freq, title=r"Refraction angle $\delta \theta$")
plt.show()

# Define working coordinate grid
el = np.linspace(0, 90, 100)  # Elevation axis
az = np.linspace(0, 360, 100)  # Azimuth axis
el_m, az_m = np.meshgrid(el, az)  # Rectangular coordinate grid

# Access refraction and attenuation in numeric form
refr, atten, _ = model.raytrace(el_m, az_m, freq)

print(f"Attenuation at {freq} MHz\n" +
      f"Min:\t{np.min(atten):.2f}\n" +
      f"Max:\t{np.max(atten):.2f}\n")

print(f"Refraction at {freq} MHz\n" +
      f"Min:\t{np.min(refr):.2f}\n" +
      f"Max:\t{np.max(refr):.2f}")