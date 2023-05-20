# importing libraries
import os
import psutil
import DarkNews as dn
import fastbnb as fb
import pandas as pd


# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory() / (1024.0**3)
        result = func(*args, **kwargs)
        mem_after = process_memory() / (1024.0**3)
        print(
            f"consumed memory: {func.__name__} --> {mem_before:.3g} GB, {mem_after:.3g} GB, {mem_after - mem_before:.3g} GB"
        )

        return result

    return wrapper


@profile
def test_gen(neval=1000):
    # 1. SIMULATIONS
    MB_fhc_df = dn.GenLauncher(
        experiment="miniboone_fhc",
        loglevel="ERROR",
        neval=neval,
    ).run()
    MB_fhc_df_dirt = dn.GenLauncher(
        experiment="miniboone_fhc_dirt",
        loglevel="ERROR",
        neval=neval,
    ).run()

    # 2. LIFETIME OF HNL
    df = dn.MC.get_merged_MC_output(MB_fhc_df, MB_fhc_df_dirt)
    # df = pd.concat([MB_fhc_df, MB_fhc_df_dirt], axis=0).reset_index(drop=True)

    df_decay = fb.decayer.decay_selection(
        df,
        l_decay_proper_cm=MB_fhc_df_dirt.attrs["N4_ctau0"],
        experiment="miniboone",
    )

    df_anal = fb.analysis.reco_nueCCQElike_Enu(df_decay, clean_df=True)

    return MB_fhc_df


test_gen(neval=1e5)
