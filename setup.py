#!/usr/bin/env python3

# proceed as usual
from setuptools import setup, Extension
import numpy as np
import os

setup_args = dict(
    # ext_modules = cythonize(["src/DarkNews/Cfourvec.pyx"]),
    # ext_modules=extensions,
    include_dirs=np.get_include(),
)


if __name__ == "__main__":
    setup(**setup_args)
