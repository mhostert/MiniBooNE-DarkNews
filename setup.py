#!/usr/bin/env python3

# proceed as usual
from setuptools import setup
import numpy as np

setup_args = dict(
    include_dirs=np.get_include(),
)


if __name__ == "__main__":
    setup(**setup_args)
