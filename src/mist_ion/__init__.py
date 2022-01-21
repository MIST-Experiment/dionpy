import warnings
from .IonModel import IonModel, srange, n_f, refr_angle, check_latlon
# from .FProfile import FProfile

try:
    from .IonModelMPI import IonModelMPI
except ImportError:
    warnings.warn(
        f"Looks like you don\'t have MPI installed. You may want to check "
        f"OpenMPI or MPICH for Linux, or Microsoft MPI for Windows.",
        RuntimeWarning,
        stacklevel=2,
    )