import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
import shutil
import uproot
# Set the colour palette to "bright"
sns.set_palette("bright")
if not os.path.exists("Figure MCFM"):
    os.makedirs("Figure MCFM")
if not os.path.exists("Result MCFM"):
    os.makedirs("Result MCFM")
# sudo umount -f /Volumes/wmass ; sudo mkdir -p /Volumes/wmass ; sudo sshfs -o reconnect -o follow_symlinks -o allow_other cyue@lxplus8s10.cern.ch:/home/c/cyue /Volumes/wmass/
# make -j256 ; export OMP_STACKSIZE=512000 ; ./mcfm input_Z.ini
# loginctl enable-linger
# systemd-run --scope --user screen -S myscreenSession

# General information
nproc = "31" # 31 (Z), 1 (W+), 6(W-)
part = "resonlyN3LO" # nlo, nnlo, resLO, resonlyNNLO, resNLO, resNLOp, resNNLO, resNNLOp, resonlyNNLOp, resonlyN3LO
# most accurate: resonlyNNLO, resNLOp, resNNLO, resonlyN3LO
ewcorrs = ["none", "sudakov", "exact"]
ewcorr = "exact"
pgoals = ["0.01", "0.005", "0.001"]
pgoal = "0.01"
name = "_".join([nproc, part, ewcorr, pgoal])

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

# # Get the command to copy the file to /eos/home-c/cyue
# suffix = "_1.0E-4" if ewcorr == "none" and part == "nnlo" or part == "nnlo" else ""
# original_file_pT = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_fine.txt"
# original_file_eta = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_y34.txt"

# original_files = [f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_{i}.txt" for i in range(3, 43)]

# copy_commands = [f"cp {file} /eos/home-c/cyue ;" for file in original_files]
# copy_commands.extend([f"cp {original_file_pT} /eos/home-c/cyue ;", f"cp {original_file_eta} /eos/home-c/cyue ;"])
# print("cd Z13TeV ;", " ".join(copy_commands), "cd ..")

# # Copy the transverse momentum data and rapidity data to the destination folder with the new names
# shutil.copy("/eos/home-c/cyue/" + original_file_pT, "Result MCFM/Z_data_{}_pT.txt".format(name))
# shutil.copy("/eos/home-c/cyue/" + original_file_eta, "Result MCFM/Z_data_{}_eta.txt".format(name))

# Extract data from text file
def extract(file):
    # Open the text file
    with open(file, "r") as f:
        # Skip the first 5 rows
        for i in range(5):
            f.readline()
        # Extract the data
        data = [line.split() for line in f.readlines()]
        if file.endswith("pT.txt") or any(file.endswith(f"pT_{i}.txt") for i in range(3, 12+1)):
            edges = np.append(np.array([float(row[0]) for row in data]), 100.0)
        elif file.endswith("eta.txt"):
            edges = np.append(np.array([float(row[0]) for row in data]), 2.5)
        values = np.array([float(row[2])/1000 for row in data])/np.diff(edges)
        errors = np.array([float(row[3])/1000 for row in data])/np.diff(edges)
    return edges, values, errors 

# Plot cross section vs transverse momentum or rapidity
def plot(file):
    if not file.startswith(".sys."):
        if file.endswith("pT.txt") or file.endswith("eta.txt"):
            file = "Result MCFM/" + file
            fig = plt.figure(figsize=(8, 6))
            if "NNPDF31" in file or "MSHT20" in file:
                lhapdf = file.split("_")[2]
                part = file.split("_")[4]
                name = file[len(f"Result MCFM/Z_data_{lhapdf}_"):-len(".txt")]
            else:
                part = file.split("_")[3]
                name = file[len(f"Result MCFM/Z_data_"):-len(".txt")]
            label = part_dict[part]
            edges, values, errors = extract(file)
            edges_centres = (edges[:-1] + edges[1:]) / 2
            yerr = errors
            plt.hist(edges[:-1], bins=edges, weights=values, histtype="step", edgecolor="black", label=name)
            plt.errorbar(edges_centres, values, yerr=yerr, fmt="none", ecolor="black", capsize=2)
            if file.endswith("pT.txt"):
                plt.xlabel(r"$p_T^Z$ (GeV)")
                plt.ylabel(r"$\sigma$ (pb)")
                plt.title(fr"$\sigma$ vs $p_T^Z$ for {label}")
                plt.legend()
                plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
                plt.savefig(f"Figure MCFM/sigma vs pT for {name}.png")
                plt.show()
            elif file.endswith("eta.txt"):
                plt.xlabel(r"$|Y^Z|$")
                plt.ylabel(r"$\sigma$ (pb)")
                plt.title(fr"$\sigma$ vs $|Y^Z|$ for {label}")
                plt.legend()
                plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
                plt.savefig(f"Figure MCFM/sigma vs abs(Y) for {name}.png")
                plt.show()


for file in os.listdir("Result MCFM"):
    plot(file)

# Plot cross section vs transverse momentum with varying electroweak correction
def plot_ewcorr_pT(ewcorrs, pgoal):
    nproc, part = "31", "nlo" # nlo, resNLOp, resNNLO, resonlyN3LO
    print(f"The precision goal is {pgoal}.")
    data_none = None
    fig = plt.figure(figsize=(8, 6))
    for ewcorr in ewcorrs:
        name = "_".join([nproc, part, ewcorr, pgoal])
        file = "Result MCFM/Z_data_{}_pT.txt".format(name)
        edges, values, errors = extract(file)
        edge_centres = (edges[:-1] + edges[1:]) / 2
        if ewcorr == "none":
            data_none = values
        else:
            yerr = (sum(values)-values)/(sum(values))**2*errors
            ratios = (values/sum(values))/(data_none/sum(data_none))
            max_deviation = max(abs(ratios-1))
            print(f"The electroweak correction is {ewcorr}, it gives a maximum deviation of {max_deviation:.2e}.")
            ratios = np.append(ratios, ratios[-1])
            plt.step(edges, ratios, where="post", linestyle = "-", label=f"{ewcorr}")
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel("ewcorr/none")
    plt.title(fr"ewcorr/none vs $p_T^Z$ for {part}, precisiongoal = {pgoal}")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.savefig(f"Figure MCFM/ratio vs pT for {nproc}_{part}_{pgoal} with varying ewcorr.png")
    plt.show()
    
for pgoal in pgoals:
    plot_ewcorr_pT(ewcorrs, pgoal)

# Plot cross section vs rapidity with varying electroweak correction
def plot_ewcorr_eta(ewcorrs, pgoal):
    nproc, part = "31", "nlo" # nlo, resNLOp, resNNLO, resonlyN3LO
    print(f"The precision goal is {pgoal}.")
    data_none = None
    fig = plt.figure(figsize=(8, 6))
    for ewcorr in ewcorrs:
        name = "_".join([nproc, part, ewcorr, pgoal])
        file = "Result MCFM/Z_data_{}_eta.txt".format(name)
        edges, values, errors = extract(file)
        edge_centres = (edges[:-1] + edges[1:]) / 2
        if ewcorr == "none":
            data_none = values
        else:
            yerr = (sum(values)-values)/(sum(values))**2*errors
            ratios = (values/sum(values))/(data_none/sum(data_none))
            max_deviation = max(abs(ratios-1))
            print(f"The electroweak correction is {ewcorr}, it gives a maximum deviation of {max_deviation:.2e}.")
            ratios = np.append(ratios, ratios[-1])
            plt.step(edges, ratios, where="post", linestyle = "-", label=f"{ewcorr}")
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel("ewcorr/none")
    plt.title(fr"ewcorr/none vs $|Y^Z|$ for {part}, precisiongoal = {pgoal}")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.savefig(f"Figure MCFM/ratio vs abs(Y) for {nproc}_{part}_{pgoal} with varying ewcorr.png")
    plt.show()
    
for pgoal in pgoals:
    plot_ewcorr_eta(ewcorrs, pgoal)

# Plot cross section vs transverse momentum with varying precision goal
def plot_pgoal_pT(pgoals, ewcorr):
    nproc, part = "31", "nlo" # nlo
    print(f"The electroweak correction is {ewcorr}.")
    data_001 = None
    fig = plt.figure(figsize=(8, 6))
    for pgoal in pgoals:
        name = "_".join([nproc, part, ewcorr, pgoal])
        file = "Result MCFM/Z_data_{}_pT.txt".format(name)
        edges, values, errors = extract(file)
        edge_centres = (edges[:-1] + edges[1:]) / 2
        if pgoal == "0.01":
            data_001 = values
        else:
            ratios = (values/sum(values))/(data_001/sum(data_001))
            yerr = (sum(values)-values)/(sum(values))**2*errors
            max_deviation = max(abs(ratios-1))
            print(f"The precision gaol is {pgoal}, it gives a maximum deviation of {max_deviation:.2e}.")
            ratios = np.append(ratios, ratios[-1])
            plt.step(edges, ratios, where="post", linestyle = "-", label=f"{ewcorr}")
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel("precisiongoal/0.01")
    plt.title(fr"precisiongoal/0.01 vs $p_T^Z$ for {part}, ewcorr = {ewcorr}")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.savefig(f"Figure MCFM/ratio vs pT for {nproc}_{part}_{ewcorr} with varying pgoal.png")
    plt.show()
    
for ewcorr in ewcorrs:
    plot_pgoal_pT(pgoals, ewcorr)

# Plot cross section vs rapidity with varying precision goal
def plot_pgoal_eta(pgoals, ewcorr):
    nproc, part = "31", "nlo" # nlo
    print(f"The electroweak correction is {ewcorr}.")
    data_001 = None
    fig = plt.figure(figsize=(8, 6))
    for pgoal in pgoals:
        name = "_".join([nproc, part, ewcorr, pgoal])
        file = "Result MCFM/Z_data_{}_eta.txt".format(name)
        edges, values, errors = extract(file)
        edge_centres = (edges[:-1] + edges[1:]) / 2
        if pgoal == "0.01":
            data_001 = values
        else:
            ratios = (values/sum(values))/(data_001/sum(data_001))
            yerr = (sum(values)-values)/(sum(values))**2*errors
            max_deviation = max(abs(ratios-1))
            print(f"The precision gaol is {pgoal}, it gives a maximum deviation of {max_deviation:.2e}.")
            ratios = np.append(ratios, ratios[-1])
            plt.step(edges, ratios, where="post", linestyle = "-", label=f"{ewcorr}")
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel("precisiongoal/0.01")
    plt.title(fr"precisiongoal/0.01 vs $|Y^Z|$ for {part}, ewcorr = {ewcorr}")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.savefig(f"Figure MCFM/ratio vs abs(Y) for {nproc}_{part}_{ewcorr} with varying pgoal.png")
    plt.show()
    
for ewcorr in ewcorrs:
    plot_pgoal_eta(pgoals, ewcorr)

# Get psuedo data
datafile = "/eos/home-c/cyue/ZMassDileptonCombineInput_xnorm.root"
data = uproot.open(datafile)
nominal = "Zmumu/xnorm_Zmumu_inclusive;1"

# Luminosity
lum = 16.8*1000

# Convert data to 2D histogram
def h2d(variable):
    return data[variable].to_hist()

# Define the bins
bins_pT, bins_eta = h2d(nominal).axes.edges
bins_pT = bins_pT.flatten()
bins_eta = bins_eta.flatten()
bin_widths_pT = np.diff(bins_pT).reshape((-1, 1))
bin_widths_eta = np.diff(bins_eta)
bin_widths = (bin_widths_eta * bin_widths_pT).flatten()

# Define the colours
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Flatten the histogram to a 1D array
def flatten(variable):
    return h2d(variable).values().flatten()/lum

# Get pseudo data
data_Zmumu = flatten(nominal)

# Select files to be compared
def sel_files(parts_cp):
    files_cp_pT = []
    files_cp_eta = []
    for part in parts_cp:
        files_cp_pT.append("Result MCFM/Z_data_31_" + part + "_exact_0.01_pT.txt")
        files_cp_eta.append("Result MCFM/Z_data_31_" + part + "_exact_0.01_eta.txt")
    return files_cp_pT, files_cp_eta
parts_cp = ["resonlyNNLO", "resNLOp", "resNNLO", "resonlyN3LO"] # resNNLOp bug
# parts_cp = ["nlo", "nnlo", "resonlyNNLO", "resNLOp", "resNNLOp", "resonlyN3LO"]
# parts_cp = ["nlo", "nnlo", "resLO", "resonlyNNLO", "resNLO", "resNLOp", "resNNLO", "resNNLOp", "resonlyNNLOp", "resonlyN3LO"]

# Compare transverse momentum plots
def compare_pT(parts_cp, pT_low):
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    hist = h2d(nominal).project("xaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    edges_Zmumu_centre = (edges_Zmumu[:-1] + edges_Zmumu[1:]) / 2
    ax1.scatter(edges_Zmumu_centre, values_Zmumu, s=25, label="data", color=colors[0])
    for i, file in enumerate(sel_files(parts_cp)[0]):
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part]
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        ax1.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values/values_Zmumu, values[-1]/values_Zmumu[-1])
        ratios_errors = np.append(yerr/values_Zmumu, yerr[-1]/values_Zmumu[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$p_T^Z$ (GeV)")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $p_T^Z$")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_Zmumu[0], edges_Zmumu[-1] if pT_low is None else pT_low)
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$p_T^Z}$ (GeV)")
    ax2.set_ylabel("Pred./Data")
#     ax2.legend()
    ax2.set_xlim(edges_Zmumu[0], edges_Zmumu[-1] if pT_low is None else pT_low)
    plt.tight_layout()
    plt.savefig(f"Figure MCFM/sigma vs pT{'' if pT_low is None else pT_low} for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()

compare_pT(parts_cp, None)

# Compare rapidity plots
def compare_eta(parts_cp):
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    hist = h2d(nominal).project("yaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    edges_Zmumu_centre = (edges_Zmumu[:-1] + edges_Zmumu[1:]) / 2
    ax1.scatter(edges_Zmumu_centre, values_Zmumu, s=25, label="data", color=colors[0])
    for i, file in enumerate(sel_files(parts_cp)[1]):
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part]
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        ax1.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values/values_Zmumu, values[-1]/values_Zmumu[-1])
        ratios_errors = np.append(yerr/values_Zmumu, yerr[-1]/values_Zmumu[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$|Y^Z|$")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $|Y^Z|$")
    ax1.legend(loc="lower left")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_Zmumu[0], edges_Zmumu[-1])
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$|Y^Z|$")
    ax2.set_ylabel("Pred./Data")
#     ax2.legend()
    ax2.set_xlim(edges_Zmumu[0], edges_Zmumu[-1])
    plt.tight_layout()
    plt.savefig(f"Figure MCFM/sigma vs abs(Y) for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()

compare_eta(parts_cp)

# Compare transverse momentum plots with pseudo data, have pT low cut
def compare_ratio_pT_low(parts_cp):
    pT_low = 30
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    hist = h2d(nominal).project("xaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    edges_Zmumu = edges_Zmumu[edges_Zmumu <= pT_low]
    values_Zmumu = values_Zmumu[:len(edges_Zmumu)-1]
    edges_Zmumu_centre = (edges_Zmumu[:-1] + edges_Zmumu[1:]) / 2
    for file in sel_files(parts_cp)[0]:
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part]
        edges, values, errors = extract(file)
        edges = edges[edges <= pT_low]
        values = values[:len(edges)-1]
        errors = errors[:len(edges)-1]
        ratios = values/values_Zmumu
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors/values_Zmumu
        ratios = np.append(ratios, ratios[-1])
        plt.step(edges, ratios, where="post", linestyle = "-", label=label)
    plt.axhline(y=1, color="black", linestyle="--")
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel("Pred./Data")
    plt.title(r"Pred./Data vs $p_T^Z$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.show()

compare_ratio_pT_low(parts_cp)

# Compare rapidity plots with pseudo data
def compare_ratio_eta(parts_cp):
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    hist = h2d(nominal).project("yaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    edges_Zmumu_centre = (edges_Zmumu[:-1] + edges_Zmumu[1:]) / 2
    for file in sel_files(parts_cp)[1]:
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part]
        edges, values, errors = extract(file)
        ratios = values/values_Zmumu
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors/values_Zmumu
        ratios = np.append(ratios, ratios[-1])
        plt.step(edges, ratios, where="post", linestyle="-", label=label)
    plt.axhline(y=1, color="black", linestyle="--")
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel("Pred./Data")
    plt.title(r"Pred./Data vs $|Y^Z|$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.savefig(f"Figure MCFM/ratio vs abs(Y) for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()

compare_ratio_eta(parts_cp)

# Compare normalized rapidity plots with pseudo data
def compare_norm_eta(parts_cp):
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    hist = h2d(nominal).project("yaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    values_Zmumu_norm = values_Zmumu/sum(values_Zmumu)
    edges_Zmumu_centres = (edges_Zmumu[:-1] + edges_Zmumu[1:]) / 2
    values_Zmumu_norm = np.append(values_Zmumu_norm, values_Zmumu_norm[-1])
    plt.step(edges_Zmumu, values_Zmumu_norm, where="post", linestyle="-", label="data")
    for file in sel_files(parts_cp)[1]:
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part]
        edges, values, errors = extract(file)
        values_norm = values/sum(values)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = (sum(values)-values)/(sum(values))**2*errors      
        values_norm = np.append(values_norm, values_norm[-1])
        plt.step(edges, values_norm, where="post", linestyle="-", label=label)
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel(r"Normalized $\sigma$")
    plt.title(r"Normalized $\sigma$ vs $|Y^Z|$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.show()

compare_norm_eta(parts_cp)

# Find maximum deviation from transverse momentum plots
def max_deviation_pT(file):
    pT_low = 30
    hist = h2d(nominal).project("xaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    edges_Zmumu = edges_Zmumu[edges_Zmumu <= pT_low]
    values_Zmumu = values_Zmumu[:len(edges_Zmumu)-1]
    edges, values, errors = extract(file)
    edges = edges[edges <= pT_low]
    values = values[:len(edges)-1]
    errors = errors[:len(edges)-1]
    ratios = values/values_Zmumu
    deviation_array = (abs(values/values_Zmumu-1))
    return max(deviation_array)

# Find maximum deviation from rapidity plots
def max_deviation_eta(file):
    hist = h2d(nominal).project("yaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    edges, values, errors = extract(file)
    values = values[:len(edges)-1]
    errors = errors[:len(edges)-1]
    ratios = values/values_Zmumu
    deviation_array = (abs(values/values_Zmumu-1))
    return max(deviation_array)

# Find the topmost maximum deviations from transverse momentum plots
def top_max_deviation_pT(parts_cp):
    files_cp_pT = sel_files(parts_cp)[0]
    files = sorted(files_cp_pT, key=lambda file: max_deviation_pT(file), reverse=True)[:int(len(files_cp_pT))]
    for file in files:
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part].replace("$\mathrm{N}^3$", "N3").replace("$\mathrm{N}^4$", "N4")
        print(fr"{label} ({part}) gives a maximum deviation of {max_deviation_pT(file):.2e}.")
top_max_deviation_pT(parts_cp)


# Find the topmost maximum deviations from rapidity plots
def top_max_deviation_eta(parts_cp):
    files_cp_eta = sel_files(parts_cp)[1]
    files = sorted(files_cp_eta, key=lambda file: max_deviation_eta(file), reverse=True)[:int(len(files_cp_eta))]
    for file in files:
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part].replace("$\mathrm{N}^3$", "N3").replace("$\mathrm{N}^4$", "N4")
        print(fr"{label} ({part}) gives a maximum deviation of {max_deviation_eta(file):.2e}.")
top_max_deviation_eta(parts_cp)

# Find chi squared value from transverse momentum plots, have pT low cut
def chi_sq_pT(file):
    pT_low = 30
    hist = h2d(nominal).project("xaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    edges_Zmumu = edges_Zmumu[edges_Zmumu <= pT_low]
    values_Zmumu = values_Zmumu[:len(edges_Zmumu)-1]
    edges, values, errors = extract(file)
    edges = edges[edges <= pT_low]
    values = values[:len(edges)-1]
    errors = errors[:len(edges)-1]
    chi_sq = ((values_Zmumu-values)**2/errors**2).sum()/len(values)
    return chi_sq

# Find chi squared value from rapidity plots
def chi_sq_eta(file):
    hist = h2d(nominal).project("yaxis")
    values_Zmumu, edges_Zmumu = hist.to_numpy()
    values_Zmumu = values_Zmumu/(edges_Zmumu[1:]-edges_Zmumu[:-1])/lum
    values_Zmumu = values_Zmumu[:len(edges_Zmumu)-1]
    edges, values, errors = extract(file)
    values = values[:len(edges)-1]
    errors = errors[:len(edges)-1]
    chi_sq = ((values_Zmumu-values)**2/errors**2).sum()/len(values)
    return chi_sq

# Find the lease chi squared value from transverse momentum plots
def least_chi_sq_pT(parts_cp):
    files_cp_pT = sel_files(parts_cp)[0]
    files = sorted(files_cp_pT, key=lambda file: chi_sq_pT(file), reverse=False)[:int(len(files_cp_pT))]
    for file in files:
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part].replace("$\mathrm{N}^3$", "N3").replace("$\mathrm{N}^4$", "N4")
        print(fr"{label} ({part}) gives a reduced chi_squared of {chi_sq_pT(file):.2e}.")
least_chi_sq_pT(parts_cp)

# Find the lease chi squared value from rapidity plots
def least_chi_sq_eta(parts_cp):
    files_cp_eta = sel_files(parts_cp)[1]
    files = sorted(files_cp_eta, key=lambda file: chi_sq_eta(file), reverse=False)[:int(len(files_cp_eta))]
    for file in files:
        part = file.split("_")[3]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        label = part_dict[part].replace("$\mathrm{N}^3$", "N3").replace("$\mathrm{N}^4$", "N4")
        print(fr"{label} ({part}) gives a reduced chi_squared of {chi_sq_eta(file):.2e}.")
least_chi_sq_eta(parts_cp)

# General information
nproc = "31" # 31 (Z), 1 (W+), 6(W-)
part = "resonlyN3LO" # nlo, nnlo, resLO, resonlyNNLO, resNLO, resNLOp, resNNLO, resNNLOp, resonlyNNLOp, resonlyN3LO
# most accurate: resonlyNNLO, resNLOp, resNNLO, resonlyN3LO
ewcorrs = ["none", "sudakov", "exact"]
ewcorr = "exact"
pgoals = ["0.01", "0.005", "0.001"]
pgoal = "0.01"
name = "_".join([nproc, part, ewcorr, pgoal])

# Get the command to copy the file to /eos/home-c/cyue
original_files = []
for i in range(3, 12+1):
    if ewcorr == "none":
        original_files.append(f"Z_only_{part}_NNPDF31_nnlo_as_0118_1.00_1.00_Z13TeV_pt34_{i}.txt")
    elif part == "nnlo":
        original_files.append(f"Z_only_{part}_{ewcorr}_NNPDF31_nnlo_as_0118_1.00_1.00_1.0E-4_Z13TeV_pt34_{i}.txt")
    else:
        original_files.append(f"Z_only_{part}_{ewcorr}_NNPDF31_nnlo_as_0118_1.00_1.00_Z13TeV_pt34_{i}.txt")
print("cd Z13TeV ;", end=" ")
for file in original_files:
    print(f"cp {file} /eos/home-c/cyue ;", end=" ")
print("cd ..")

# # Copy the transverse momentum data and rapidity data to the destination folder with the new names
# i = 3
# for file in original_files:
#     shutil.copy("/eos/home-c/cyue/" + file, f"Result MCFM/Z_data_{name}_pT_{i}.txt")
#     i += 1


# Parts used to generate 2D histogram
parts_2d = ["resonlyNNLO", "resNLOp", "resNNLO", "resonlyN3LO"]

# Generate 2D histogram
def hist_2d(part):
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    label = part_dict[part]
    name = "_".join([nproc, part, ewcorr, pgoal])
    files = []
    values_array = []
    for i in range(3, 12+1):
        files.append(f"Z_data_{name}_pT_{i}.txt")
    for file in files:
        file = "Result MCFM/" + file
        edges, values, errors = extract(file)
        values_array.append(values)
    x_edges, y_edges = np.meshgrid(bins_pT, bins_eta)
    z = values_array
    plt.pcolormesh(x_edges, y_edges, z, cmap=plt.cm.gray_r)
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel(r"$|Y^Z|$")
    plt.title(fr"2D distribution of $\sigma$ for {part}")
    plt.colorbar(label=r"$\sigma$ (pb)")
    plt.text(0.95, 0.95, label, transform=ax.transAxes, ha="right", va="top", fontsize=12)
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, ha="right", va="top")
    plt.show()
for part in parts_2d:
    hist_2d(part)

# Extract data from text file
def extract_bin(file):
    # Open the text file
    with open(file, "r") as f:
        # Skip the first 5 rows
        for i in range(5):
            f.readline()
        # Extract the data
        data = [line.split() for line in f.readlines()]
        if any(file.endswith(f"pT_{i}.txt") for i in range(3, 12+1)):
            edges = np.append(np.array([float(row[0]) for row in data]), 100.0)
            values = np.array([float(row[2])/1000 for row in data])
        errors = np.array([float(row[3])/1000 for row in data])
    return edges, values, errors 

# Plot variable vs data
def plot_vs_data_Zmumu(parts_2d):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    ax1.plot(np.arange(len(data_Zmumu)), data_Zmumu/bin_widths, label=r"$Z\to\mu\mu$", color=colors[0])
    for j, part in enumerate(parts_2d):
        nproc, ewcorr, pgoal = "31", "exact", "0.01"
        label = part_dict[part]
        name = "_".join([nproc, part, ewcorr, pgoal])
        files = []
        values_array = []
        errors_array = []
        for i in range(3, 12+1):
            files.append(f"Z_data_{name}_pT_{i}.txt")
        for file in files:
            file = "Result MCFM/" + file
            edges, values, errors = extract_bin(file)
            values_array.append(values)
            errors_array.append(errors)
        values = np.array(values_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        errors = np.array(errors_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        ax1.plot(np.arange(len(values)), values/bin_widths, linestyle=":", label=f"{label}", color=colors[j+1])
        ratios = values/data_Zmumu
        ax2.plot(np.arange(len(values)), ratios, "-", label=f"{label}", color=colors[j+1])
    ax1.set_xlabel("bin")
    ax1.set_ylabel(r"$\sigma$/bin")
    ax1.set_title(r"$\sigma$ vs bin")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel("bin")
    ax2.set_ylabel("Pred./Data")
#     ax2.legend()
    plt.tight_layout()
    plt.savefig(f"Figure MCFM/sigma vs bin for {name}.png")
    plt.show()

plot_vs_data_Zmumu(parts_2d)

# Plot variable vs data
def plot_ratio_data_Zmumu(parts_2d):
    fig = plt.figure(figsize=(8, 6))
    for j, part in enumerate(parts_2d):
        nproc, ewcorr, pgoal = "31", "exact", "0.01"
        label = part_dict[part]
        name = "_".join([nproc, part, ewcorr, pgoal])
        files = []
        values_array = []
        errors_array = []
        for i in range(3, 12+1):
            files.append(f"Z_data_{name}_pT_{i}.txt")
        for file in files:
            file = "Result MCFM/" + file
            edges, values, errors = extract_bin(file)
            values_array.append(values)
            errors_array.append(errors)
        values = np.array(values_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        errors = np.array(errors_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        errors = errors/data_Zmumu
        ratios = values/data_Zmumu
        plt.plot(np.arange(len(ratios)), ratios, linestyle=":", label=f"{label}", color=colors[j+1])
    plt.axhline(y=1, linestyle="--", label="y = 1", color=colors[0])
    plt.xlabel("bin")
    plt.ylabel("Pred./Data")
    plt.title("Pred./Data vs bin")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.savefig(f"Figure MCFM/ratio vs bin for {name}.png")
    plt.show()

plot_ratio_data_Zmumu(parts_2d)

# Find chi squared value from 2D bins plot, have pT low cut
def chi_sq(part, pT_low):
    edges_pT = bins_pT
    edges_eta = bins_eta
    if pT_low == None:
        cut = None
    else:
        cut = (len([x for x in edges_pT if x <= pT_low])-1)*(len(edges_eta)-1)
    values_Zmumu = data_Zmumu[:cut]
    nproc, ewcorr, pgoal = "31", "none", "0.001"
    label = part_dict[part]
    name = "_".join([nproc, part, ewcorr, pgoal])
    files = []
    values_array = []
    errors_array = []
    for part in parts_2d:
        nproc, ewcorr, pgoal = "31", "exact", "0.01"
        label = part_dict[part]
        name = "_".join([nproc, part, ewcorr, pgoal])
        files = []
        values_array = []
        errors_array = []
        for i in range(3, 12+1):
            files.append(f"Z_data_{name}_pT_{i}.txt")
        for file in files:
            file = "Result MCFM/" + file
            edges, values, errors = extract_bin(file)
            values_array.append(values)
            errors_array.append(errors)
    values = np.array(values_array).T.flatten()[:cut] # Transpose the 2D array and flatten it into a 1D array
    errors = np.array(errors_array).T.flatten()[:cut] # Transpose the 2D array and flatten it into a 1D array
    with np.errstate(divide="ignore"):
        chi_sq = ((values_Zmumu-values)**2/errors**2)[((values_Zmumu-values)**2/errors**2) <= 1e4].sum()/len(values)
    return chi_sq

# Find the lease chi squared value from 2D bins plot, have pT low cut
def least_chi_sq(parts, pT_low):
    parts = sorted(parts, key=lambda part: chi_sq(part, pT_low), reverse=False)[:int(len(parts))]
    for part in parts:
        print(f"{part} gives a reduced chi_squared of {chi_sq(part, pT_low):.2e}.")

least_chi_sq(parts_2d, None)
least_chi_sq(parts_2d, 30)
