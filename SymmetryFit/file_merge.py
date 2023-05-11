import argparse
import glob
import os
from fastbnb import fit_functions as ff
from fastbnb import grid_fit as gf

FILE_TYPES = [
    "chi2",
    "evis_spectrum",
    "enu_spectrum",
    "costheta_spectrum",
    "invmass_spectrum",
]


def merge_files(path):
    for file_type in FILE_TYPES:
        outfilename = path + f"{file_type}.dat"
        for filename in glob.glob(f"{path}{file_type}*.dat"):
            if filename == outfilename:
                # don't want to copy the output into the output
                continue
            with open(filename, "r") as readfile:
                gf.save_to_locked_file(outfilename, str(readfile.read()))

        for filename in glob.glob(f"{path}{file_type}*.dat"):
            if filename == outfilename:
                # don't want to remove the output file
                continue
            os.remove(filename)


######## Input from command line ########
parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="path to the simulation results")
args = parser.parse_args()

######## Running fit function ########
merge_files(args.path)
