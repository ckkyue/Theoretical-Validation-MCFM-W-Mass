import h5py
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.linalg import cho_solve, cho_factor, inv
import seaborn as sns
import shutil
import uproot
datafile = "Result MCFM Real Data/fitresult.hdf5"
prefitfile = "Result MCFM Real Data/fitresult_prefit.hdf5"
# Set the colour palette to "bright"
sns.set_palette("bright")
folders = ["Result MCFM Real Data", "Figure MCFM Real Data"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
# sudo umount -f /Volumes/wmass ; sudo mkdir -p /Volumes/wmass ; sudo sshfs -o reconnect -o follow_symlinks -o allow_other cyue@lxplus8s10.cern.ch:/home/c/cyue /Volumes/wmass/
# make -j256 ; export OMP_STACKSIZE=512000 ; ./mcfm input_Z.ini
# loginctl enable-lingerin
# systemd-run --scope --user screen -S Session0
# ctrl+A+D

# General information
nproc = "31" # 31 (Z), 1 (W+), 6(W-)
part = "resNLOp" # resNNLOp bug
# nnlo, resLO, resonlyNNLO, resNLO, resNLOp, resNNLO, resNNLOp, resonlyNNLOp, resonlyN3LO
# most accurate: resonlyNNLO, resNLOp, resNNLO, resonlyN3LO
lhapdfs = ["NNPDF31", "MSHT20"] # NNPDF31 or MSHT20
ewcorrs = ["none", "sudakov", "exact"]
ewcorr = "none"
pgoals = ["0.01", "0.005", "0.001"]
pgoal = "0.001"
name = "_".join([nproc, part, ewcorr, pgoal])

# Select the files to be compared
def sel_files(parts_cp):
    files_cp_pT = []
    files_cp_eta = []
    for part in parts_cp:
        for lhapdf in lhapdfs:
            files_cp_pT.append(f"Result MCFM Real Data/Z_data_{lhapdf}_31_" + part + "_none_0.001_pT.txt")
            files_cp_eta.append(f"Result MCFM Real Data/Z_data_{lhapdf}_31_" + part + "_none_0.001_eta.txt")
    return files_cp_pT, files_cp_eta
def check_pgoal(files):
    for i, file in enumerate(files):
        if not file.startswith("Result MCFM Real Data/"):
            file = "Result MCFM Real Data/" + file
        if os.path.exists(file):
            continue
        else:
            files[i] = file.replace("0.001", "0.01")
    return files
parts_cp = ["resNLOp", "resNNLO"]

# Description of part
part_dict = {
    "nlo": "NLO",
    "nnlo": "NNLO",
    "resLO": "NLL resummed and matched",
    "resonlyLO": "NLL resummed only",
    "resonlyLOp": "NLLp resummed only",
    "resexpNLO": "NNLL resummed expanded to NLO",
    "resonlyNLO": "NNLL resummed only",
    "resaboveNLO": "Fixed-order matching to NLO",
    "resmatchcorrNLO": "Matching corrections at NLO",
    "resonlyNLOp": "NNLLp resummed only",
    "resexpNNLO": r"$\mathrm{N}^3$LL resummed expanded to NNLO",
    "resonlyNNLO": r"$\mathrm{N}^3$LL resummed only",
    "resaboveNNLO": "Fixed-order matching to NNLO",
    "resLOp": "NLLp resummed and matched",
    "resNLO": "NNLL resummed, matched to NLO",
    "resNLOp": r"$\mathrm{N}^3$LL resummed, matched to NLO",
    "resNNLO": r"$\mathrm{N}^3$LL resummed, matched to NNLO",
    "resNNLOp": r"$\mathrm{N}^3$LLp resummed, matched to NNLO",
    "resonlyNNLOp": r"$\mathrm{N}^3$LLp resummed only",
    "resonlyN3LO": r"$\mathrm{N}^4$LL resummed only"
}

# Get the command to copy the file to /eos/home-c/cyue
suffix = "_1.0E-4" if ewcorr == "none" and part == "nnlo" or part == "nnlo" else ""
original_file_pT = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_fine.txt"
original_file_eta = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_y34.txt"

original_files = [f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_{i}.txt" for i in range(3, 12+1)]

copy_commands = [f"cp {file} /eos/home-c/cyue ;" for file in original_files]
copy_commands.extend([f"cp {original_file_pT} /eos/home-c/cyue ;", f"cp {original_file_eta} /eos/home-c/cyue ;"])
print("cd Z13TeV ;", " ".join(copy_commands), "cd ..")

# # Get the command to copy the file to /eos/home-c/cyue
# suffix = "_1.0E-4" if ewcorr == "none" and part == "nnlo" or part == "nnlo" else ""
# original_file_pT = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}MSHT20nnlo_as118_1.00_1.00{suffix}_Z13TeV_pt34_fine.txt"
# original_file_eta = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}MSHT20nnlo_as118_1.00_1.00{suffix}_Z13TeV_y34.txt"

# original_files = [f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}MSHT20nnlo_as118_1.00_1.00{suffix}_Z13TeV_pt34_{i}.txt" for i in range(3, 12+1)]

# copy_commands = [f"cp {file} /eos/home-c/cyue ;" for file in original_files]
# copy_commands.extend([f"cp {original_file_pT} /eos/home-c/cyue ;", f"cp {original_file_eta} /eos/home-c/cyue ;"])
# print("cd Z13TeV ;", " ".join(copy_commands), "cd ..")

# # Copy the transverse momentum data and rapidity data to the destination folder with the new names
# shutil.copy("/eos/home-c/cyue/" + original_file_pT, "Result MCFM Real Data/Z_data_NNPDF31_{}_pT.txt".format(name))
# shutil.copy("/eos/home-c/cyue/" + original_file_eta, "Result MCFM Real Data/Z_data_NNPDF31_{}_eta.txt".format(name))
# i = 3
# for file in original_files:
#     shutil.copy("/eos/home-c/cyue/" + file, f"Result MCFM Real Data/Z_data_NNPDF31_{name}_pT_{i}.txt")
#     i += 1
# print(i)

# # Copy the transverse momentum data and rapidity data to the destination folder with the new names
# shutil.copy("/eos/home-c/cyue/" + original_file_pT, "Result MCFM Real Data/Z_data_MSHT20_{}_pT.txt".format(name))
# shutil.copy("/eos/home-c/cyue/" + original_file_eta, "Result MCFM Real Data/Z_data_MSHT20_{}_eta.txt".format(name))
# i = 3
# for file in original_files:
#     shutil.copy("/eos/home-c/cyue/" + file, f"Result MCFM Real Data/Z_data_MSHT20_{name}_pT_{i}.txt")
#     i += 1
# print(i)

# Define the bins
bins_pT = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 27, 32, 40, 55, 100])
bins_eta = np.arange(0, 2.75, 0.25)
bin_widths_pT = np.diff(bins_pT).reshape((-1, 1))
bin_widths_eta = np.diff(bins_eta)
bin_widths = (bin_widths_eta * bin_widths_pT).flatten()

# Define the colours
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Extract data from text file
def extract(file):
    if not file.startswith(".sys."):
        if file.endswith(".hdf5"):
            file = h5py.File(file, "r")
            values, errors = zip(*file["data_ZptVGen_absYVGen"])
            values, errors = np.array(values), np.sqrt(np.array(errors))
            values_pT = np.cumsum(values.reshape(-1, 10), axis=1)[:, -1]/np.diff(bins_pT)
            values_eta = np.sum(values.reshape(20, 10), axis=0)/np.diff(bins_eta)
            errors_pT = np.cumsum(errors.reshape(-1, 10), axis=1)[:, -1]/np.diff(bins_pT)
            errors_eta = np.sum(errors.reshape(20, 10), axis=0)/np.diff(bins_eta)
            return values_pT, values_eta, errors_pT, errors_eta
        else:
            # Open the text file
            with open(file, "r") as f:
                if file.endswith("pT.txt") or any(file.endswith(f"pT_{i}.txt") for i in range(3, 12+1)) or file.endswith("eta.txt"):
                    # Skip the first 5 rows
                    for i in range(5):
                        f.readline()
                    # Extract the data
                    data = [line.split() for line in f.readlines()]
                    if file.endswith("pT.txt") or any(file.endswith(f"pT_{i}.txt") for i in range(3, 42+1)):
                        edges = np.append(np.array([float(row[0]) for row in data]), 100.0)
                    else:
                        edges = np.append(np.array([float(row[0]) for row in data]), 5.0)
                    values = np.array([float(row[2])/1000 for row in data])/np.diff(edges)
                    errors = np.array([float(row[3])/1000 for row in data])/np.diff(edges)
                    return edges, values, errors

# Plot cross section vs transverse momentum or rapidity
def plot(file):
    if not file.startswith(".sys."):
        if file.endswith("pT.txt") or file.endswith("eta.txt"):
            file = "Result MCFM Real Data/" + file
            fig = plt.figure(figsize=(8, 6))
            if "NNPDF31" in file or "MSHT20" in file:
                lhapdf = file.split("_")[2]
                part = file.split("_")[4]
                name = file[len(f"Result MCFM Real Data/Z_data_{lhapdf}_"):-len(".txt")]
            else:
                part = file.split("_")[3]
                name = file[len(f"Result MCFM Real Data/Z_data_"):-len(".txt")]
            label = f"{part_dict[part]} ({lhapdf}) "
            edges, values, errors = extract(file)
            edges_centres = (edges[:-1] + edges[1:]) / 2
            yerr = errors
            plt.hist(edges[:-1], bins=edges, weights=values, histtype="step", edgecolor="black", label=label)
            plt.errorbar(edges_centres, values, yerr=yerr, fmt="none", ecolor="black", capsize=2)
            if file.endswith("pT.txt"):
                plt.xlabel(r"$p_T^Z$ (GeV)")
                plt.ylabel(r"$\sigma$ (pb)")
                plt.title(fr"$\sigma$ vs $p_T^Z$")
                plt.legend()
                plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
                plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
                plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
                plt.savefig(f"Figure MCFM Real Data/sigma vs pT for {name}.png")
                plt.show()
            elif file.endswith("eta.txt"):
                plt.xlim(0, 2.5)
                plt.xlabel(r"$|Y^Z|$")
                plt.ylabel(r"$\sigma$ (pb)")
                plt.title(fr"$\sigma$ vs $|Y^Z|$")
                plt.legend()
                plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
                plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
                plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
                plt.savefig(f"Figure MCFM Real Data/sigma vs abs(Y) for {name}.png")
                plt.show()
                
for file in os.listdir("Result MCFM Real Data"):
    plot(file)

# Compare transverse momentum plots
def compare_pT(parts_cp, pT_low):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    values_pT, values_eta, errors_pT, errors_eta = extract(datafile)
    edges_pT = bins_pT
    edges_pT_centres = (edges_pT[:-1] + edges_pT[1:]) / 2
    label = r"unfolded $Z\to\mu\mu$ data"
    ax1.scatter(edges_pT_centres, values_pT, label=label, s=20, color="black")
    ax1.errorbar(edges_pT_centres, values_pT, yerr=errors_pT, fmt="none", ecolor="black", capsize=2)
    values_pT_fit, values_eta_fit, errors_pT_fit, errors_eta_fit = extract(prefitfile)
    label = r"SCETlib + DYTurbo $\otimes$ MiNNLO + PHOTOS++"
    ax1.hist(edges_pT[:-1], bins=edges_pT, weights=values_pT_fit, histtype="step", label=label, color=colors[0])
    ax1.fill_between(edges_pT, np.append(values_pT_fit-errors_pT_fit, values_pT_fit[-1]-errors_pT_fit[-1]), np.append(values_pT_fit+errors_pT_fit, values_pT_fit[-1]+errors_pT_fit[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    ratios_fit = np.append(values_pT_fit/values_pT, values_pT_fit[-1]/values_pT[-1])
    ratios_errors_fit = np.append(errors_pT_fit/values_pT, errors_pT_fit[-1]/values_pT[-1])
    ax2.step(edges_pT, ratios_fit, where="post", linestyle="-", label=label, color=colors[0])
    ax2.fill_between(edges_pT, ratios_fit-ratios_errors_fit, ratios_fit+ratios_errors_fit, alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_pT = check_pgoal(sel_files(parts_cp)[0])
    for i, file in enumerate(files_cp_pT):
        lhapdf = file.split("_")[2]
        part = file.split("_")[4]
        name = file[len(f"Result MCFM Real Data/Z_data_{lhapdf}_"):-len(".txt")]
        label = part_dict[part] + f" ({lhapdf})"
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        ax1.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values/values_pT, values[-1]/values_pT[-1])
        ratios_errors = np.append(yerr/values_pT, yerr[-1]/values_pT[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$p_T^Z$ (GeV)")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $p_T^Z$")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.set_xlabel(r"$p_T^Z$ (GeV)")
    ax2.set_ylabel("Pred./Data")
#     ax2.legend()
    ax2.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
#     ax2.set_ylim(0.8, 1.1)
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Real Data/sigma vs pT{'' if pT_low is None else pT_low} for real data {len(parts_cp)} sets Z.png")

compare_pT(parts_cp, None)
compare_pT(parts_cp, 20)

# Compare normalized transverse momentum plots
def compare_norm_pT(parts_cp, pT_low):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    values_pT, values_eta, errors_pT, errors_eta = extract(datafile)
    values_pT_norm = values_pT/sum(values_pT)
    edges_pT = bins_pT
    edges_pT_centres = (edges_pT[:-1] + edges_pT[1:]) / 2
    yerr = (sum(values_pT)-values_pT)/(sum(values_pT))**2*errors_pT
    label = r"unfolded $Z\to\mu\mu$ data"
    ax1.scatter(edges_pT_centres, values_pT_norm, label=label, s=20, color="black")
    ax1.errorbar(edges_pT_centres, values_pT_norm, yerr=yerr, fmt="none", ecolor="black", capsize=2)
    values_pT_fit, values_eta_fit, errors_pT_fit, errors_eta_fit = extract(prefitfile)
    values_pT_fit_norm = values_pT_fit/sum(values_pT_fit)
    errors_pT_fit_norm = (sum(values_pT_fit)-values_pT_fit)/(sum(values_pT_fit))**2*errors_pT_fit
    label = r"SCETlib + DYTurbo $\otimes$ MiNNLO + PHOTOS++"
    ax1.hist(edges_pT[:-1], bins=edges_pT, weights=values_pT_fit_norm, histtype="step", label=label, color=colors[0])
    ax1.fill_between(edges_pT, np.append(values_pT_fit_norm-errors_pT_fit_norm, values_pT_fit_norm[-1]-errors_pT_fit_norm[-1]), np.append(values_pT_fit_norm+errors_pT_fit_norm, values_pT_fit_norm[-1]+errors_pT_fit_norm[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    ratios_fit = np.append(values_pT_fit_norm/values_pT_norm, values_pT_fit_norm[-1]/values_pT_norm[-1])
    ratios_errors_fit = np.append(errors_pT_fit_norm/values_pT_norm, errors_pT_fit_norm[-1]/values_pT_norm[-1])
    ax2.step(edges_pT, ratios_fit, where="post", linestyle="-", label=label, color=colors[0])
    ax2.fill_between(edges_pT, ratios_fit-ratios_errors_fit, ratios_fit+ratios_errors_fit, alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_pT = check_pgoal(sel_files(parts_cp)[0])
    for i, file in enumerate(files_cp_pT):
        lhapdf = file.split("_")[2]
        part = file.split("_")[4]
        name = file[len(f"Result MCFM Real Data/Z_data_{lhapdf}_"):-len(".txt")]
        label = part_dict[part] + f" ({lhapdf})"
        edges, values, errors = extract(file)
        values_norm = values/sum(values)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = (sum(values)-values)/(sum(values))**2*errors        
        ax1.hist(edges[:-1], bins=edges, weights=values_norm, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values_norm-yerr, values_norm[-1]-yerr[-1]), np.append(values_norm+yerr, values_norm[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values_pT_norm/values_norm, values_pT_norm[-1]/values_norm[-1])
        ratios_errors = np.append(yerr/values_norm, yerr[-1]/values_norm[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$p_T^Z$ (GeV)")
    ax1.set_ylabel(r"Normalized $\sigma$")
    ax1.set_title(r"Normalized $\sigma$ vs $p_T^Z$")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax2.axhline(1.0, linestyle="--", color="black")
    ax1.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
    ax2.set_xlabel(r"$p_T^Z$ (GeV)")
    ax2.set_ylabel("Pred./Data")
    ax2.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
    ax2.set_ylim(0.9, 1.15)
#     ax2.legend()
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Real Data/Normalized sigma vs pT{'' if pT_low is None else pT_low} for real data {len(parts_cp)} sets Z.png")
    plt.show()

compare_norm_pT(parts_cp, None)
compare_norm_pT(parts_cp, 20)

# Compare rapidity plots
def compare_eta(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    values_pT, values_eta, errors_pT, errors_eta = extract(datafile)
    edges_eta = bins_eta
    edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
    label = r"unfolded $Z\to\mu\mu$ data"
    ax1.scatter(edges_eta_centres, values_eta, label=label, s=20, color="black")
    ax1.errorbar(edges_eta_centres, values_eta, yerr=errors_eta, fmt="none", ecolor="black", capsize=2)
    values_pT_fit, values_eta_fit, errors_pT_fit, errors_eta_fit = extract(prefitfile)
    label = r"SCETlib + DYTurbo $\otimes$ MiNNLO + PHOTOS++"
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta_fit, histtype="step", label=label, color=colors[0])
    ax1.fill_between(edges_eta, np.append(values_eta_fit-errors_eta_fit, values_eta_fit[-1]-errors_eta_fit[-1]), np.append(values_eta_fit+errors_eta_fit, values_eta_fit[-1]+errors_eta_fit[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    ratios_fit = np.append(values_eta_fit/values_eta, values_eta_fit[-1]/values_eta[-1])
    ratios_errors_fit = np.append(errors_eta_fit/values_eta, errors_eta_fit[-1]/values_eta[-1])
    ax2.step(edges_eta, ratios_fit, where="post", linestyle="-", label=label, color=colors[0])
    ax2.fill_between(edges_eta, ratios_fit-ratios_errors_fit, ratios_fit+ratios_errors_fit, alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_eta = check_pgoal(sel_files(parts_cp)[1])
    for i, file in enumerate(files_cp_eta):
        lhapdf = file.split("_")[2]
        part = file.split("_")[4]
        name = file[len(f"Result MCFM Inclusive/Z_data_{lhapdf}_"):-len(".txt")]
        label = part_dict[part] + f" ({lhapdf})"
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        ax1.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values/values_eta, values[-1]/values_eta[-1])
        ratios_errors = np.append(yerr/values_eta, yerr[-1]/values_eta[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$|Y^Z|$")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $|Y^Z|$")
    ax1.legend(loc="lower left")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(0, 2.5)
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.set_xlabel(r"$|Y^Z|$")
    ax2.set_ylabel("Pred./Data")
#     ax2.legend()
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0.8, 1.2)
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Real Data/sigma vs abs(Y) for real data {len(parts_cp)} sets Z.png")
    plt.show()

compare_eta(parts_cp)

# Compare normalized rapidity plots with real data
def compare_norm_eta(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    values_pT, values_eta, errors_pT, errors_eta = extract(datafile)
    values_eta_norm = values_eta/sum(values_eta)
    edges_eta = bins_eta
    edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
    yerr = (sum(values_eta)-values_eta)/(sum(values_eta))**2*errors_eta
    label = r"unfolded $Z\to\mu\mu$ data"
    ax1.scatter(edges_eta_centres, values_eta_norm, label=label, s=20, color="black")
    ax1.errorbar(edges_eta_centres, values_eta_norm, yerr=yerr, fmt="none", ecolor="black", capsize=2)
    values_pT_fit, values_eta_fit, errors_pT_fit, errors_eta_fit = extract(prefitfile)
    values_eta_fit_norm = values_eta_fit/sum(values_eta_fit)
    errors_eta_fit_norm = (sum(values_eta_fit)-values_eta_fit)/(sum(values_eta_fit))**2*errors_eta_fit
    label = r"SCETlib + DYTurbo $\otimes$ MiNNLO + PHOTOS++"
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta_fit_norm, histtype="step", label=label, color=colors[0])
    ax1.fill_between(edges_eta, np.append(values_eta_fit_norm-errors_eta_fit_norm, values_eta_fit_norm[-1]-errors_eta_fit_norm[-1]), np.append(values_eta_fit_norm+errors_eta_fit_norm, values_eta_fit_norm[-1]+errors_eta_fit_norm[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    ratios_fit = np.append(values_eta_fit_norm/values_eta_norm, values_eta_fit_norm[-1]/values_eta_norm[-1])
    ratios_errors_fit = np.append(errors_eta_fit_norm/values_eta_norm, errors_eta_fit_norm[-1]/values_eta_norm[-1])
    ax2.step(edges_eta, ratios_fit, where="post", linestyle="-", label=label, color=colors[0])
    ax2.fill_between(edges_eta, ratios_fit-ratios_errors_fit, ratios_fit+ratios_errors_fit, alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_eta = check_pgoal(sel_files(parts_cp)[1])
    for i, file in enumerate(files_cp_eta):
        lhapdf = file.split("_")[2]
        part = file.split("_")[4]
        name = file[len(f"Result MCFM Real Data/Z_data_{lhapdf}_"):-len(".txt")]
        label = part_dict[part] + f" ({lhapdf})"
        edges, values, errors = extract(file)
        values_norm = values/sum(values)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = (sum(values)-values)/(sum(values))**2*errors        
        ax1.hist(edges[:-1], bins=edges, weights=values_norm, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values_norm-yerr, values_norm[-1]-yerr[-1]), np.append(values_norm+yerr, values_norm[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values_eta_norm/values_norm, values_eta_norm[-1]/values_norm[-1])
        ratios_errors = np.append(yerr/values_norm, yerr[-1]/values_norm[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$|Y^Z|$")
    ax1.set_ylabel(r"Normalized $\sigma$")
    ax1.set_title(r"Normalized $\sigma$ vs $|Y^Z|$")
    ax1.legend(loc="lower left")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(0, 2.5)
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.set_xlabel(r"$|Y^Z|$")
    ax2.set_ylabel("Pred./Data")
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0.9, 1.1)
#     ax2.legend()
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Real Data/Normalized sigma vs abs(Y) for real data {len(parts_cp)} sets Z.png")
    plt.show()

compare_norm_eta(parts_cp)

# Extract data from text file
def extract_bin(file):
    # Open the text file
    if file.endswith(".hdf5"):
        file = h5py.File(file, "r")
        values, errors = zip(*file["data_ZptVGen_absYVGen"])
        values, errors = np.array(values), np.sqrt(np.array(errors))
        return values, errors
    else:
        with open(file, "r") as f:
            if any(file.endswith(f"pT_{i}.txt") for i in range(3, 12+1)):
                # Skip the first 5 rows
                for i in range(5):
                    f.readline()
                # Extract the data
                data = [line.split() for line in f.readlines()]
                edges = np.append(np.array([float(row[0]) for row in data]), 100.0)
                values = np.array([float(row[2])/1000 for row in data])
                errors = np.array([float(row[3])/1000 for row in data])
                return edges, values, errors

# Plot variable vs real data
def plot_vs_data(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    values_Zmumu, errors_Zmumu = extract_bin(datafile)
    label = r"unfolded $Z\to\mu\mu$ data"
    ax1.scatter(np.arange(len(values_Zmumu)), values_Zmumu/bin_widths, label=label, s=10, color="black")
    ax1.errorbar(np.arange(len(values_Zmumu)), values_Zmumu/bin_widths, yerr=errors_Zmumu, fmt="none", ecolor="black", capsize=2)
    values_Zmumu_fit, errors_Zmumu_fit = extract_bin(prefitfile)
    label = r"SCETlib + DYTurbo $\otimes$ MiNNLO + PHOTOS++"
    ax1.plot(np.arange(len(values_Zmumu_fit)), values_Zmumu_fit/bin_widths, linestyle="-", label=label, color=colors[0])
    ratios_fit = values_Zmumu_fit/values_Zmumu
    ax2.plot(np.arange(len(values_Zmumu_fit)), ratios_fit, "-", label="label", color=colors[0])
    next(ax1._get_lines.prop_cycler)
    next(ax2._get_lines.prop_cycler)
    for lhapdf in lhapdfs:
        for part in parts_cp:
            nproc, ewcorr, pgoal = "31", "none", "0.001"
            label = part_dict[part] + f" ({lhapdf})"
            name = "_".join([nproc, part, ewcorr, pgoal])
            files = []
            values_array = []
            errors_array = []
            for i in range(3, 12+1):
                files.append(f"Z_data_{lhapdf}_{name}_pT_{i}.txt")
            files = check_pgoal(files)
            for file in files:
                if not file.startswith("Result MCFM Real Data/"):
                    file = "Result MCFM Real Data/" + file
                edges, values, errors = extract_bin(file)
                values_array.append(values)
                errors_array.append(errors)
            values = np.array(values_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
            errors = np.array(errors_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
            ax1.plot(np.arange(len(values)), values/bin_widths, linestyle=":", label=f"{label}")
            ratios = values/values_Zmumu
            ax2.plot(np.arange(len(values)), ratios, "-", label=f"{label}")
    ax1.set_xlabel("bin")
    ax1.set_ylabel(r"$d\sigma$/bin")
    ax1.set_title(r"$\sigma$ vs bin")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax2.axhline(1.0, linestyle="--", color="black")
    ax2.set_ylim(0, 2)
    ax2.set_xlabel("bin")
    ax2.set_ylabel("Pred./Data")
#     ax2.legend()
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Real Data/sigma vs bin for real data {len(parts_cp)} sets Z.png")
    plt.show()

plot_vs_data(parts_cp)

# Luminosity
lum = 16.8*1000

# Find chi squared value from 2D bins plot, have pT low cut
def chi_sq(part, lhapdf, pT_low):
    edges_pT = bins_pT
    edges_eta = bins_eta
    if pT_low == None:
        cut = None
    else:
        cut = (len([x for x in edges_pT if x <= pT_low])-1)*(len(edges_eta)-1)
    values_Zmumu, errors_Zmumu = extract_bin(datafile)
    values_Zmumu, errors_Zmumu = values_Zmumu[:cut], errors_Zmumu[:cut]
    nproc, ewcorr, pgoal = "31", "none", "0.001"
    label = part_dict[part]
    name = "_".join([nproc, part, ewcorr, pgoal])
    files = []
    values_array = []
    errors_array = []
    for i in range(3, 12+1):
        files.append(f"Z_data_{lhapdf}_{name}_pT_{i}.txt")
    files = check_pgoal(files)
    for file in files:
        if not file.startswith("Result MCFM Real Data/"):
            file = "Result MCFM Real Data/" + file
        edges, values, errors = extract_bin(file)
        values_array.append(values)
        errors_array.append(errors)
    values = np.array(values_array).T.flatten()[:cut] # Transpose the 2D array and flatten it into a 1D array
    errors = np.array(errors_array).T.flatten()[:cut] # Transpose the 2D array and flatten it into a 1D array
    K_MCFM = np.diag(errors**2)
    K_data = np.array(h5py.File(datafile, "r")["covariance_matrix_ZptVGen_absYVGen"])[:cut, :cut]/lum**2
    diff = (values_Zmumu-values)
    chi_sq = (diff.T@cho_solve(cho_factor(K_data+K_MCFM), diff))/len(values)
#     chi_sq = diff.T@inv(K_data+K_MCFM)@diff/len(values)
#     chi_sq = diff.T@np.linalg.inv(K_data+K_MCFM)@diff/len(values)
    return chi_sq

# Find the lease chi squared value from 2D bins plot, have pT low cut
def least_chi_sq(parts, lhapdf, pT_low):
    parts = sorted(parts, key=lambda part: chi_sq(part, lhapdf, pT_low), reverse=False)[:int(len(parts))]
    for part in parts:
        print(f"{part} ({lhapdf}) gives a reduced chi_squared of {chi_sq(part, lhapdf, pT_low):.2e}.")

import time
start = time.time()
least_chi_sq(parts_cp, "NNPDF31", None)
least_chi_sq(parts_cp, "NNPDF31", 30)
least_chi_sq(parts_cp, "MSHT20", None)
least_chi_sq(parts_cp, "MSHT20", 30)
end = time.time()
print(end-start)
