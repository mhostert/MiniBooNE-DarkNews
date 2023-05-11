from pathlib import Path
import numpy as np
from datetime import datetime
from slurmpter import Slurm, SlurmJob


def get_unique_path(path):
    # Appends a unique time string to the path to avoid overwriting previous results
    Path(path).mkdir(parents=True, exist_ok=True)  # create it if it doesn't exist
    now = datetime.now()  # Time stamp
    now_string = f"{now.month}m_{now.day}d_{now.hour}h_{now.minute}m_{now.second}s"
    return f"{path}{now_string}_"  # path to all result files for a given job submission


def slurm_submit(
    fit_script,
    path,
    XGRID,
    YGRID,
    input_kwargs,
    jobname="dnfitter",
    optional_args="",
    queue="debugq",
    exclude="cn070",
    timeout="01:00:00",
):
    if queue == "debugq":
        print(
            "Watchoutu for the timeout you provided. Is it less thna 1 h? You specified",
            timeout,
        )
    ####################################################
    # Create input file for the scan
    if np.shape(XGRID) != np.shape(YGRID):
        raise ValueError("XGRID and YGRID must have the same shape.")
    # if np.shape(XGRID)[0] != np.shape(XGRID)[1]:
    # raise ValueError("XGRID and YGRID must be square grids.")

    xpoints = np.shape(XGRID)[1]
    ypoints = np.shape(XGRID)[0]
    scanfile = path + "input_scan.dat"
    with open(scanfile, "wb") as f:
        for i, (x, y) in enumerate(zip(XGRID.flatten(), YGRID.flatten())):
            f.write(f"{x} {y}\n".encode())

    # save input dictionary to file
    kfile = path + "input_kwargs.npy"
    with open(kfile, "wb") as f:
        np.save(kfile, input_kwargs)

    ####################################################
    """
        We now submit jobs using JobArray function of SLURM. This is far more efficient when using the defq. 
        And it does not clutter the queue.
        Essentially, there's only one job, which keeps sending jobs to open cores/nodes as they become available.
        The (x, y) values are passed to the job as an array index (SLURM_ARRAY_TASK_ID), 
        which is then used to read the corresponding values from the scanfile.
    """
    # Define the job directories to write submit files, standard outputs and standard errors.
    submit = "slurm/submit"
    output = "slurm/output"
    error = "slurm/error"
    Path(submit).mkdir(parents=True, exist_ok=True)
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(error).mkdir(parents=True, exist_ok=True)

    slurm = Slurm(
        name=jobname,
        submit=submit,
        extra_lines=[
            f"#SBATCH --partition={queue}",
            f"#SBATCH --exclude={exclude}",
        ],
    )

    ####################################################
    job_generate = SlurmJob(
        name=slurm.name,
        executable="python",
        # nodes=1,
        extra_srun_options=["ntasks=1"],
        extra_lines=[
            f"#SBATCH --partition={queue}",
            f"#SBATCH --exclude={exclude}",
            f"#SBATCH --ntasks {xpoints-1}",
            f"#SBATCH --time={timeout}",
            # f"#SBATCH -N 1",
            # f"#SBATCH --cpus-per-task=1",
            f"#SBATCH --array=0-{ypoints-1}",
            # f"#SBATCH --mem-per-cpu=3gb",
            # f"#SBATCH --ntasks-per-core=1",
        ],
        submit=submit,
        output=output,
        error=error,
        slurm=slurm,
    )
    for i in range(xpoints):
        job_generate.add_arg(
            f"{fit_script} --i_line $( expr \"$SLURM_ARRAY_TASK_ID\" '*' {xpoints} '+' {i} ) --path {path} "
            + optional_args
        )

    ####################################################
    # Instantiate a SlurmJob object for mergin all the data in this batch
    job_merge = SlurmJob(
        name="merge-data",
        executable="python",
        extra_lines=[f"#SBATCH --partition={queue}", f"#SBATCH --exclude={exclude}"],
        submit=submit,
        output=output,
        error=error,
        slurm=slurm,
    )
    job_merge.add_parent(job_generate)
    job_merge.add_arg(f"file_merge.py {path}")

    ####################################################
    """
        Job submission
    """
    print(f"Submitting jobs... Your path is: {path}")
    print("\nFollow the path.\n")

    slurm.build_submit()
