import warnings
from .IonModel import IonModel, srange, n_f, refr_angle, check_latlon, nu_p, nu_p_warm
from .FProfile import FProfile

from .scripts.build_iri import build_iri

try:
    from .IonModelMPI import IonModelMPI
except ImportError:
    warnings.warn(
        f"Looks like you don\'t have MPI installed. You may want to check "
        f"OpenMPI or MPICH for Linux, or Microsoft MPI for Windows.",
        RuntimeWarning,
        stacklevel=2,
    )