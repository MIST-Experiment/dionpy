Installation
============

..
    The ``dionpy`` package provides model of ionosphere refraction and attenuation based
    on the `IRI2016 Ionosphere Model <https://irimodel.org/>`_.
    The `Memo 62 <http://www.physics.mcgill.ca/mist/memos/MIST_memo_62.pdf>`_ of the
    `MIST <http://www.physics.mcgill.ca/mist/>`_ experiment introduced a new Python interface to the IRI 2016 Fortran core
    code - the ``iricore`` package. This interface was optimized for the MIST purposes.
    The ``iricore`` will be installed automatically during the installation of the ``dionpy`` if you meet all the requirements.
    However, the ``iricore`` can be compiled only on Linux systems, which puts the same restriction on the ``dionpy``.
    If you are using Windows - consider installing `WSL <https://docs.microsoft.com/en-us/windows/wsl/install>`_.

The ``dionpy`` can be compiled only on Linux systems.
If you are using Windows - consider installing `WSL <https://docs.microsoft.com/en-us/windows/wsl/install>`_.

Before installing the ``dionpy`` pakage you need preinstalled:

* Fortran compiler, e.g. `GFortran <https://gcc.gnu.org/wiki/GFortran>`_
* `CMAKE <https://cmake.org/>`_
* `FFmpeg <https://ffmpeg.org/>`_ - for rendering animated models

.. code-block::

    sudo apt install gfortran cmake ffmpeg

Now you can simply install ``dionpy`` using ``pip`` or any other package manager::

    python3 -m pip install dionpy


