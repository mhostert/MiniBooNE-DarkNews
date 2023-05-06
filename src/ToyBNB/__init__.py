__version__ = "0.0.1"

"""
    Initializing loggers
"""
import logging

# for debug and error handling
toy_logger = logging.getLogger(__name__ + ".logger")

# Analysis and plotting modules
from ToyBNB import analysis
from ToyBNB import decayer
from ToyBNB import fastmc
from ToyBNB import plot_tools
