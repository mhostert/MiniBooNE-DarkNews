__version__ = "0.0.1"

"""
    Initializing loggers
"""
import logging

# for debug and error handling
toy_logger = logging.getLogger(__name__ + ".logger")

# Analysis and plotting modules 
from ToyBNB import analysis
from ToyBNB import analysis_decay
from ToyBNB import cuts
from ToyBNB import plot_tools