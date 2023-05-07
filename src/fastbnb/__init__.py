__version__ = "0.0.1"

"""
    Initializing loggers
"""
import logging

# for debug and error handling
toy_logger = logging.getLogger(__name__ + ".logger")

# Analysis and plotting modules
from fastbnb import analysis
from fastbnb import decayer
from fastbnb import fastmc
from fastbnb import plot_tools
