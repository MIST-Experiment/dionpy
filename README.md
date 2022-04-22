# mist_ion
A fork of [mist_ionosphere](https://github.com/erika-hornecker/mist_ionosphere). 
Mist_ion provides tools for ionosphere modeling (refraction angles, attenuation factors, etc.)
for the [MIST](http://www.physics.mcgill.ca/mist/) project. A more detailed overview will be 
available soon.

## Installing
Prerequisites
- MPI implementation, for example
  - [Open MPI](https://www.open-mpi.org/) 
  - [MPICH](https://www.mpich.org/)
  - [Microsoft MPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (for Windows)

Then you can install the package via pip
```
python3 -m pip install mistion
```

[//]: # (## Quick example)

[//]: # (```python)

[//]: # (from mist_ion import IonModel)

[//]: # (from datetime import datetime)

[//]: # ()
[//]: # (# Date and time of observation)

[//]: # (dt = datetime.strptime&#40;'2012-08-15 06:00', '%Y-%m-%d %H:%M'&#41;)

[//]: # ()
[//]: # (# Initial model parameters: telescope position, frequency of observations, datetime)

[//]: # (model = IonModel&#40;)

[//]: # (    lat0=79.433,)

[//]: # (    lon0=-90.766,)

[//]: # (    alt0=0.,)

[//]: # (    freq=45e6,)

[//]: # (    dt=dt)

[//]: # (&#41;)

[//]: # ()
[//]: # (# Create a grid of [gridsize x gridsize] points - azimuth and elevation &#40;telescope POV&#41;)

[//]: # (model.generate_coord_grid&#40;gridsize=100&#41;)

[//]: # ()
[//]: # (# Setup D layer and F layer parameters &#40;number of sublayers, lower and upper limits in m&#41;)

[//]: # (model.setup_dlayer&#40;nlayers=10, d_bot = 6e4, d_top = 9e4&#41;)

[//]: # (model.setup_flayer&#40;nlayers=30&#41;)

[//]: # ()
[//]: # (# Starting calculation with 16 parallel processes)

[//]: # (model.calc&#40;progressbar=True, processes=16&#41;)

[//]: # ()
[//]: # (# Saving results for later use)

[//]: # (model.save&#40;name='test', dir='results'&#41;)

[//]: # ()
[//]: # (# Loading model)

[//]: # (new_model = IonModel.load&#40;'results/test.h5'&#41;)

[//]: # ()
[//]: # (# Plotting results)

[//]: # (model.plot&#40;data='d_e_density'&#41;)

[//]: # ()
[//]: # (# Plotting and saving results)

[//]: # (model.plot&#40;data='d_e_density', file='d_e_density_plot.png'&#41;)

[//]: # (```)