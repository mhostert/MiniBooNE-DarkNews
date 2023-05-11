from pathlib import Path
import numpy as np
from fastbnb import fit_functions as ff
from datetime import datetime
from slurmpter import Slurm, SlurmJob

##########################
"""
    Some global variables for the grid fitting procedure
"""
QUEUE = "defq"  # "defq" or "debugq"
RUN_PATH = "results/3p1_general/"  # Path for this type of scan
Path(RUN_PATH).mkdir(parents=True, exist_ok=True)  # create it if it doesn't exist

now = datetime.now()  # Time stamp
now_string = f"{now.month}m_{now.day}d_{now.hour}h_{now.minute}m_{now.second}s"
path = f"{RUN_PATH}{now_string}_"  # path to all result files for this job submission

CUT = "circ1"  # "circ0" or "circ1" or "diag1" or "invmass"
PRINT_SPECTRA = 1  # 0 -- do not print spectrum file or 1 -- print spectrum files (Enu, Evis, Cotheta, and invmass spectra.)
EXPERIMENT = "miniboone_fhc"

####################################################
"""
    Now some parameters for the event generation. 
    This is saved to file and then loaded by the fitting script.
"""
kwargs = {
    "Umu4": np.sqrt(1.0e-12),
    "UD4": 1.0 / np.sqrt(2.0),
    "gD": 2.0,
    "epsilon": 8e-4,
    "neval": int(1e5),
    "HNLtype": "dirac",
    "pandas": False,
    "parquet": False,
    "loglevel": "ERROR",
}

# save input dictionary to file
kfile = path + "input_kwargs.npy"
with open(kfile, "wb") as f:
    np.save(kfile, kwargs)

####################################################
# Initiate main output file:
cols = [
    "mzprime",
    "m4",
    "sum_w_post_smearing",
    "v_mu4",
    "v_4i",
    "epsilon",
    "u_mu4",
    "chi2",
    "decay_length",
    "N_events",
]

with open(path + "chi2.dat", "w") as f:
    # Header
    f.write("# " + " ".join(cols) + "\n")

# Define the job directories to write submit files, standard outputs and standard errors.
submit = "slurm/submit"
output = "slurm/output"
error = "slurm/error"
Path(submit).mkdir(parents=True, exist_ok=True)
Path(output).mkdir(parents=True, exist_ok=True)
Path(error).mkdir(parents=True, exist_ok=True)

####################################################
"""
    We create the grid and then print all tuples of m4,mzprime to a file.
    This file will be read by the fitting script based on a row index.
"""
points = 31  # NOTE: JobArray max size is 1001.
mzvals = np.array([0.03, 0.2, 0.8, 1])  # ---> This is fixed -- run for multiple plots
m4vals = ff.round_sig(np.geomspace(0.01, 1.0, points), 4)
XGRID, YGRID = np.meshgrid(m4vals, mzvals)
scanfile = path + "input_scan.dat"
with open(scanfile, "wb") as f:
    for i, (m4, mz) in enumerate(zip(XGRID.flatten(), YGRID.flatten())):
        f.write(f"{m4} {mz}\n".encode())

####################################################
"""
    We now submit jobs using JobArray function of SLURM. This is far more efficient when using the defq. 
    And it does not clutter the queue.
    Essentially, there's only one job, which keeps sending jobs to open cores/nodes as they become available.
    The (m4, mz) values are passed to the job as an array index (SLURM_ARRAY_TASK_ID), 
    which is then used to read the corresponding values from the scanfile.
"""
slurm = Slurm(
    name=f"3p1_coupl:.2g}",
    submit=submit,
    extra_lines=[f"#SBATCH --partition={QUEUE}"],
)

job = SlurmJob(
    name=slurm.name,
    executable="python",
    extra_srun_options=["ntasks=1"],
    extra_lines=[
        f"#SBATCH --partition={QUEUE}",
        f"#SBATCH --array=0-{(points**2 - 1)}",
    ],
    submit=submit,
    output=output,
    error=error,
    slurm=slurm,
)

####################################################
"""
    Job submission
"""
print(f"Submitting jobs... Your path is: {path}")
print("\nFollow the path.\n")
job.add_arg(
    f"fit_3p1_cooupling.py --i_line $SLURM_ARRAY_TASK_ID --path {path} --cut {CUT} --print_spectra {PRINT_SPECTRA} --experiment {EXPERIMENT}"
)
slurm.build_submit()
