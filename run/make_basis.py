import numpy as np
import h5py
import argparse
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build reduced basis from warped simulation results"
    )
    parser.add_argument("folder", type=str, help="Path to data folder")
    parser.add_argument("labels", type=str, help="Path to file containing patient data")
    parser.add_argument("sub", type=str, help="Name of target subject")
    args = parser.parse_args()

    folder = Path(args.folder)

    h5_data = folder.joinpath("data.h5")
    basis = folder.joinpath("basis.h5")

    coefs = []

    # Build patient info dataframe
    subs = []
    df_arr = []

    # Read dataframe with labels
    df = pd.read_excel(
        args.labels,
        engine="odf",
        skiprows=1,
    ).fillna(value=0)

    df = df.drop(
        columns=[
            "New ID",
            "Orignal ID",
            "Unnamed: 2",
            "New ID.1",
            "Unnamed: 5",
            "PasientID-MR",
        ]
    )
    df = df[:104]
    df = df.rename(columns={"Orignal ID.1": "Sub"})
    df["Sub"] = df["Sub"].apply(lambda x: "Sub-" + str(x)[-3:])
    df_sub_list = df["Sub"].to_list()

    with h5py.File(h5_data, "r") as f:
        g = f["warped"]

        for k, v in g.items():
            vals = v["coefs"][()]

            if (
                not np.any(np.isnan(vals))
                and not np.any(np.isinf(vals))
                and not np.all(
                    np.isclose(v["coefs"][()], np.zeros_like(v["coefs"][()]))
                )
                and k in df_sub_list
            ):
                coefs.append(vals)
                # Add row for each warped brain
                dd = {}
                dd["Sub"] = k
                dd["target"] = False
                dd["L2"] = v["L2"][()]
                dd["H1"] = v["H1"][()]
                dd["hausdorff"] = v["hausdorff"][()]
                for (name,compval) in v["compartments"].items():
                    dd[name + "_L2"] = compval["L2"][()] 
                    dd[name + "_H1"] = compval["H1"][()] 
                df_arr.append(dd)
                subs.append(k)
            else:
                print("Invalid values. Skipping...")

    # Add row for the target state
    target_dd = df_arr[-1].copy()
    for k, v in target_dd.items():
        target_dd[k] = 0
    target_dd["Sub"] = args.sub
    subs.append(args.sub)
    target_dd["target"] = True
    df_arr.append(target_dd)
    data_df = pd.DataFrame(df_arr)

    df = df[df["Sub"].isin(subs)]
    df = df.merge(data_df)
    df.to_csv(folder.joinpath("labels").with_suffix(".csv"))

    def save_basis(mat,group_name):
        U,S,Vh = np.linalg.svd(mat,full_matrices=False)
        with h5py.File(basis, "a") as f:
            for name, data in zip(["U", "S", "Vh"], [U, S, Vh]):
                try:
                    g = f.create_group(group_name)
                except:
                    g = f[group_name]
                    
                try:
                    g[name] = data
                except:
                    g[name][...] = data
    
    # Compute SVD of entire dataset
    coefs = np.array(coefs).T
    save_basis(coefs, "dataset")
    
    # Compute SVD of dataset without iNPH brains
    iids = df.index[df["iNPH"] != 1.0].to_list()[:-1] # Last entry will be the target state
    coefs_sans_iNPH = coefs[:,iids]
    save_basis(coefs_sans_iNPH, "sansiNPH")
    
    # Compute SVD of dataset with only iNPH brains
    iids = df.index[df["iNPH"] == 1.0].to_list()[:-1] # Last entry will be the target state
    coefs_iNPH = coefs[:,iids]
    save_basis(coefs_iNPH, "onlyiNPH")
