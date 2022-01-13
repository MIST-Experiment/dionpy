# mist_ion
A fork of [mist_ionosphere](https://github.com/erika-hornecker/mist_ionosphere). 
Mist_ion provides tools for ionosphere modeling (refraction angles, attenuation factors, etc.)
for the [MIST](http://www.physics.mcgill.ca/mist/) project. A more detailed overview will be 
available soon.

## Installing
You can install the package via pip
```
python3 -m pip install git+https://github.com/lap1dem/mist_ion.git
```

It will install all necessary dependencies. However, if you want to run
parallel calculations on systems with distributed memory (e. g. clusters),
you will need an MPI implementation like [Open MPI](https://www.open-mpi.org/) 
or [MPICH](https://www.mpich.org/).

## Quick example
```python
from mist_ion import IonModel
from datetime import datetime

# Date and time of observation
dt = datetime.strptime('2012-08-15 06:00', '%Y-%m-%d %H:%M')

# Initial model parameters: telescope position, frequency of observations, datetime
model = IonModel(
    lat0=79.433,
    lon0=-90.766,
    alt0=0.,
    freq=45e6,
    dt=dt
)

# Create a grid of [gridsize x gridsize] points - azimuth and elevation (telescope POV)
model.generate_coord_grid(gridsize=100)

# Setup D layer and F layer parameters (number of sublayers, lower and upper limits in m)
model.setup_dlayer(nlayers=10, d_bot = 6e4, d_top = 9e4)
model.setup_flayer(nlayers=30)

# Starting calculation with 16 parallel processes
model.calc(progressbar=True, processes=16)

# Saving results for later use
model.save(name='test', dir='results')

# Loading model
new_model = IonModel.load('results/test.h5')

# Plotting results
model.plot(data='d_e_density')

# Plotting and saving results
model.plot(data='d_e_density', file='d_e_density_plot.png')
```