import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle, lz4.frame
import re
import seaborn as sns
import shutil
import uproot
datafile = "DYTurbo z0/results_z-2d-nnlo-vj-member0-scetlibmatch.txt"
# Set the colour palette to "bright"
sns.set_palette("bright")
folders = ["Figure MCFM Inclusive", "Result MCFM Inclusive", "Figure DYTurbo z0", "Figure DYTurbo res"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
# sudo umount -f /Volumes/wmass ; sudo mkdir -p /Volumes/wmass ; sudo sshfs -o reconnect -o follow_symlinks -o allow_other cyue@lxplus8s10.cern.ch:/home/c/cyue /Volumes/wmass/
# make -j256 ; export OMP_STACKSIZE=512000 ; ./mcfm input_Z_Inclusive.ini
# loginctl enable-linger
# systemd-run --scope --user screen -S Session0
# ctrl+A+D

# General information
nproc = "31" # 31 (Z), 1 (W+), 6(W-)
part = "resNLOp" # resNNLOp bug
# nnlo, resLO, resonlyNNLO, resNLO, resNLOp, resNNLO, resNNLOp, resonlyNNLOp, resonlyN3LO
# most accurate: resonlyNNLO, resNLOp, resNNLO, resonlyN3LO
ewcorrs = ["none", "sudakov", "exact"]
ewcorr = "none"
pgoals = ["0.01", "0.005", "0.001"]
pgoal = "0.01"
name = "_".join([nproc, part, ewcorr, pgoal])

# Select files to be compared
def sel_files(parts_cp):
    files_cp_pT = []
    files_cp_eta = []
    for part in parts_cp:
        files_cp_pT.append("Result MCFM Inclusive/Z_data_31_" + part + "_none_0.001_pT.txt")
        files_cp_eta.append("Result MCFM Inclusive/Z_data_31_" + part + "_none_0.001_eta.txt")
    return files_cp_pT, files_cp_eta
def check_pgoal(files):
    for i, file in enumerate(files):
        if not file.startswith("Result MCFM Inclusive/"):
            file = "Result MCFM Inclusive/" + file
        if os.path.exists(file):
            continue
        else:
            files[i] = file.replace("0.001", "0.01")
    return files
parts_cp = ["resNLOp", "resNNLO"]
parts_2d = parts_cp

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
# shutil.copy("/eos/home-c/cyue/" + original_file_pT, "Result MCFM Inclusive/Z_data_{}_pT.txt".format(name))
# shutil.copy("/eos/home-c/cyue/" + original_file_eta, "Result MCFM Inclusive/Z_data_{}_eta.txt".format(name))
# i = 3
# for file in original_files:
#     shutil.copy("/eos/home-c/cyue/" + file, f"Result MCFM Inclusive/Z_data_{name}_pT_{i}.txt")
#     i += 1

# Define the bins
bins_pT = np.concatenate(([0.1], np.arange(1, 101)))
bins_eta = np.arange(-5, 5.25, 0.25)
bin_widths_pT = np.diff(bins_pT).reshape((-1, 1))
bin_widths_eta = np.diff(bins_eta)
bin_widths = (bin_widths_eta * bin_widths_pT).flatten()

# Define the colours
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Extract data from text file
def extract(file):
    if not file.startswith(".sys."):
        # Open the text file
        with open(file, "r") as f:
            if file.endswith("pT.txt") or any(file.endswith(f"pT_{i}.txt") for i in range(3, 42+1)) or file.endswith("eta.txt"):
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
            elif file.endswith("scetlibmatch.txt"):
                # Skip the first 1 row
                for i in range(1):
                    f.readline()
                # Extract the data
                data = [line.split() for line in f.readlines()[:-1]]
                edges_pT = np.append(np.array([float(row[2]) for row in data])[:100], 100.0)
                edges_eta = np.append(np.array([float(row[0]) for row in data])[::100], 5.0)
                values_pT = np.array([sum(np.array([float(row[4])/1000 for row in data])[i::100]) for i in range(100)])/np.diff(edges_pT)
                values_eta = np.array([sum(np.array([float(row[4])/1000 for row in data])[i*100:(i+1)*100]) for i in range(len(np.array([float(row[4])/1000 for row in data]))//100)])/np.diff(edges_eta)
                errors_pT = np.array([sum(np.array([float(row[5])/1000 for row in data])[i::100]) for i in range(100)])/np.diff(edges_pT)
                errors_eta = np.array([sum(np.array([float(row[5])/1000 for row in data])[i*100:(i+1)*100]) for i in range(len(np.array([float(row[5])/1000 for row in data]))//100)])/np.diff(edges_eta)
                return edges_pT, edges_eta, values_pT, values_eta, errors_pT, errors_eta

# Plot cross section vs transverse momentum or rapidity
def plot(file):
    if not file.startswith(".sys."):
        if file.endswith("pT.txt") or file.endswith("eta.txt"):
            file = "Result MCFM Inclusive/" + file
            fig = plt.figure(figsize=(8, 6))
            if "NNPDF31" in file or "MSHT20" in file:
                lhapdf = file.split("_")[2]
                part = file.split("_")[4]
                name = file[len(f"Result MCFM Inclusive/Z_data_{lhapdf}_"):-len(".txt")]
            else:
                part = file.split("_")[3]
                name = file[len(f"Result MCFM Inclusive/Z_data_"):-len(".txt")]
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
                plt.savefig(f"Figure MCFM Inclusive/sigma vs pT for {name}.png")
                plt.show()
            elif file.endswith("eta.txt"):
                plt.xlabel(r"$Y^Z$")
                plt.ylabel(r"$\sigma$ (pb)")
                plt.title(fr"$\sigma$ vs $Y^Z$ for {label}")
                plt.legend()
                plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
                plt.savefig(f"Figure MCFM Inclusive/sigma vs Y for {name}.png")
                plt.show()
        elif file.endswith("scetlibmatch.txt"):
            if "an3lo" in datafile:
                member = datafile.split("-")[5].replace("member", "")
                label_DYTurbo = f"DYTurbo an3lo {member}"
            else:
                member = datafile.split("-")[4].replace("member", "")
                label_DYTurbo = f"DYTurbo {member}"
            file = "DYTurbo z0/" + file
            fig = plt.figure(figsize=(8, 6))
            member = file.split("-")[4].replace("member", "")
            label = f"DYTurbo {member}"
            edges_pT, edges_eta, values_pT, values_eta, errors_pT, errors_eta = extract(file)
            edges_pT_centres = (edges_pT[:-1] + edges_pT[1:]) / 2
            edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
            yerr_pT = errors_pT
            yerr_eta = errors_eta
            plt.hist(edges_pT[:-1], bins=edges_pT, weights=values_pT, histtype="step", edgecolor="black", label=label)
            plt.errorbar(edges_pT_centres, values_pT, yerr=yerr_pT, fmt="none", ecolor="black", capsize=2)
            plt.xlabel(r"$p_T^Z$ (GeV)")
            plt.ylabel(r"$\sigma$ (pb)")
            plt.title(fr"$\sigma$ vs $p_T^Z$ for {label}")
            plt.legend()
            plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
            plt.savefig(f"Figure MCFM Inclusive/sigma vs pT for {label}.png")
            plt.show()
            fig = plt.figure(figsize=(8, 6))
            plt.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta, histtype="step", edgecolor="black", label=label)
            plt.errorbar(edges_eta_centres, values_eta, yerr=yerr_eta, fmt="none", ecolor="black", capsize=2)
            plt.xlabel(r"$Y^Z$")
            plt.ylabel(r"$\sigma$ (pb)")
            plt.title(fr"$\sigma$ vs $Y^Z$ for {label}")
            plt.legend()
            plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
            plt.savefig(f"Figure DYTurbo z0/sigma vs Y for {label}.png")
            plt.show()

for file in os.listdir("Result MCFM Inclusive"):
    plot(file)
for file in os.listdir("DYTurbo z0"):
    plot(file)

# Compare transverse momentum plots
def compare_pT(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_pT, edges_eta, values_pT, values_eta, errors_pT, errors_eta = extract(datafile)
    edges_pT_centres = (edges_pT[:-1] + edges_pT[1:]) / 2
    if "an3lo" in datafile:
        member = datafile.split("-")[5].replace("member", "")
        label_DYTurbo = f"DYTurbo an3lo {member}"
    else:
        member = datafile.split("-")[4].replace("member", "")
        label_DYTurbo = f"DYTurbo {member}"
    ax1.hist(edges_pT[:-1], bins=edges_pT, weights=values_pT, histtype="step", label=label_DYTurbo, color=colors[0])
    ax1.fill_between(edges_pT, np.append(values_pT-errors_pT, values_pT[-1]-errors_pT[-1]), np.append(values_pT+errors_pT, values_pT[-1]+errors_pT[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_pT = check_pgoal(sel_files(parts_cp)[0])
    for i, file in enumerate(files_cp_pT):
        part = file.split("_")[3]
        name = file[len("Result MCFM Inclusive/Z_data_"):-len(".txt")]
        label = part_dict[part]
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        ax1.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values/values_pT, values[-1]/values_pT[-1])
        ratios_errors = np.append(yerr/values_pT, yerr[-1]/values_pT[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$p_{T}^Z$ (GeV)")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $p_T^Z$")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_pT[0], edges_pT[-1])
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$p_{T}^Z$ (GeV)")
    ax2.set_ylabel("Pred./DYTurbo")
#     ax2.legend()
    ax2.set_xlim(edges_pT[0], edges_pT[-1])
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo z0/sigma vs pT for {label_DYTurbo} {len(parts_cp)} sets.png")
    plt.show()

compare_pT(parts_cp)

# Compare rapidity plots
def compare_eta(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_pT, edges_eta, values_pT, values_eta, errors_pT, errors_eta = extract(datafile)
    edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
    if "an3lo" in datafile:
        member = datafile.split("-")[5].replace("member", "")
        label_DYTurbo = f"DYTurbo an3lo {member}"
    else:
        member = datafile.split("-")[4].replace("member", "")
        label_DYTurbo = f"DYTurbo {member}"
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta, histtype="step", label=label_DYTurbo, color=colors[0])
    ax1.fill_between(edges_eta, np.append(values_eta-errors_eta, values_eta[-1]-errors_eta[-1]), np.append(values_eta+errors_eta, values_eta[-1]+errors_eta[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_eta = check_pgoal(sel_files(parts_cp)[1])
    for i, file in enumerate(files_cp_eta):
        part = file.split("_")[3]
        name = file[len("Result MCFM Inclusive/Z_data_"):-len(".txt")]
        label = part_dict[part]
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        ax1.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
        ax1.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        ratios = np.append(values/values_eta, values[-1]/values_eta[-1])
        ratios_errors = np.append(yerr/values_eta, yerr[-1]/values_eta[-1])
        ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
        ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
    ax1.set_xlabel(r"$Y^Z$")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $Y^Z$")
    ax1.legend(loc="lower center")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_eta[0], edges_eta[-1])
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$Y^Z$")
    ax2.set_ylabel("Pred./DYTurbo")
#     ax2.legend()
    ax2.set_xlim(edges_eta[0], edges_eta[-1])
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo z0/sigma vs Y for {label_DYTurbo} {len(parts_cp)} sets.png")
    plt.show()

compare_eta(parts_cp)

# Compare normalized rapidity plots with pseudo data
def compare_norm_eta(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_pT, edges_eta, values_pT, values_eta, errors_pT, errors_eta = extract(datafile)
    values_eta_norm = values_eta/sum(values_eta)
    edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
    yerr = (sum(values_eta)-values_eta)/(sum(values_eta))**2*errors_eta
    if "an3lo" in datafile:
        member = datafile.split("-")[5].replace("member", "")
        label_DYTurbo = f"DYTurbo an3lo {member}"
    else:
        member = datafile.split("-")[4].replace("member", "")
        label_DYTurbo = f"DYTurbo {member}"
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta_norm, histtype="step", label=label_DYTurbo, color=colors[0])
    ax1.fill_between(edges_eta, np.append(values_eta_norm-yerr, values_eta_norm[-1]-yerr[-1]), np.append(values_eta_norm+yerr, values_eta_norm[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_eta = check_pgoal(sel_files(parts_cp)[1])
    for i, file in enumerate(files_cp_eta):
        part = file.split("_")[3]
        name = file[len("Result MCFM Inclusive/Z_data_"):-len(".txt")]
        label = part_dict[part]
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
    ax1.set_xlabel(r"$Y^Z$")
    ax1.set_ylabel(r"Normalized $\sigma$")
    ax1.set_title(r"Normalized $\sigma$ vs $Y^Z$")
    ax1.legend(loc="lower center")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$Y^Z$")
    ax2.set_ylabel("Pred./DYTurbo")
#     ax2.legend()
    ax2.set_xlim(edges_eta[0], edges_eta[-1])
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo z0/Normalized sigma vs Y for {label_DYTurbo} {len(parts_cp)} sets.png")
    plt.show()

compare_norm_eta(parts_cp)

# Generate 2D histogram
def hist_2d(part):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    label = part_dict[part]
    pgoal = "0.001"
    name = "_".join([nproc, part, ewcorr, pgoal])
    files = []
    values_array = []
    for i in range(3, 42+1):
        files.append(f"Z_data_{name}_pT_{i}.txt")
    files = check_pgoal(files)
    for file in files:
        if not file.startswith("Result MCFM Inclusive/"):
            file = "Result MCFM Inclusive/" + file
        edges, values, errors = extract(file)
        values_array.append(values)
    x_edges, y_edges = np.meshgrid(bins_pT, bins_eta)
    z = values_array
    plt.pcolormesh(x_edges, y_edges, z, cmap=plt.cm.gray_r)
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel(r"$Y^Z$")
    plt.title(fr"2D distribution of $\sigma$ for {part}")
    plt.colorbar(label=r"$\sigma$ (pb)")
    plt.text(0.95, 0.95, label, transform=ax.transAxes, ha="right", va="top", fontsize=12)
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, ha="right", va="top")
    label = part_dict[part].replace("$\mathrm{N}^3$", "N3").replace("$\mathrm{N}^4$", "N4")
    plt.savefig(f"Figure MCFM Inclusive/2D distribution of sigma for {label}.png")
    plt.show()
    
for part in parts_2d:
    hist_2d(part)

# Reshape an array
def group_and_flatten(list, n):
    groups = [list[i:i+n] for i in range(0, len(list), n)]
    return np.array(groups).T.flatten()

# Extract data from text file
def extract_bin(file):
    # Open the text file
    with open(file, "r") as f:
        if any(file.endswith(f"pT_{i}.txt") for i in range(3, 42+1)):
            # Skip the first 5 rows
            for i in range(5):
                f.readline()
            # Extract the data
            data = [line.split() for line in f.readlines()]
            edges = np.append(np.array([float(row[0]) for row in data]), 100.0)
            values = np.array([float(row[2])/1000 for row in data])
            errors = np.array([float(row[3])/1000 for row in data])
        elif file.endswith("scetlibmatch.txt") and not file.startswith(".sys."):
            # Skip the first 1 row
            for i in range(1):
                f.readline()
            # Extract the data
            data = [line.split() for line in f.readlines()[:-1]]
            edges = np.append(np.array([float(row[2]) for row in data])[:100], 100.0)
            values = group_and_flatten(np.array([float(row[4])/1000 for row in data]), 100)
            errors = group_and_flatten(np.array([float(row[5])/1000 for row in data]), 100)
    return edges, values, errors

# Plot variable vs data
def plot_vs_DYTurbo(parts_2d, datafile):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_Zmumu, values_Zmumu, errors_Zmumu = extract_bin(datafile)
    if "an3lo" in datafile:
        member = datafile.split("-")[5].replace("member", "")
        label_DYTurbo = f"DYTurbo an3lo {member}"
    else:
        member = datafile.split("-")[4].replace("member", "")
        label_DYTurbo = f"DYTurbo {member}"
    ax1.plot(np.arange(len(values_Zmumu)), values_Zmumu/bin_widths, linestyle="-", label=label_DYTurbo, color=colors[0])
    for j, part in enumerate(parts_2d):
        nproc, ewcorr, pgoal = "31", "none", "0.001"
        label = part_dict[part]
        name = "_".join([nproc, part, ewcorr, pgoal])
        files = []
        values_array = []
        errors_array = []
        for i in range(3, 42+1):
            files.append(f"Z_data_{name}_pT_{i}.txt")
        files = check_pgoal(files)
        for file in files:
            if not file.startswith("Result MCFM Inclusive/"):
                file = "Result MCFM Inclusive/" + file
            edges, values, errors = extract_bin(file)
            values_array.append(values)
            errors_array.append(errors)
        values = np.array(values_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        errors = np.array(errors_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        ax1.plot(np.arange(len(values)), values/bin_widths, linestyle=":", label=f"{label}", color=colors[j+1])
        ratios = values/values_Zmumu
        ax2.plot(np.arange(len(values)), ratios, "-", label=f"{label}", color=colors[j+1])
    ax1.set_xlabel("bin")
    ax1.set_ylabel(r"$d\sigma$/bin")
    ax1.set_title(r"$\sigma$ vs bin")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel("bin")
    ax2.set_ylabel("Pred./DYTurbo")
#     ax2.legend()
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo z0/sigma vs bin for {label_DYTurbo} {len(parts_2d)} sets.png")
    plt.show()
    
plot_vs_DYTurbo(parts_2d, datafile)

# Get the file of DYTurbo resummation
home = "/eos/home-c/cyue"
res_path = "TheoryCorrections/scetlib_dyturboCorrZ.pkl.lz4"
res_file = os.path.join(home, res_path)
res = pickle.load(lz4.frame.open(res_file))["Z"]["scetlib_dyturbo_hist"][{"vars": "pdf0"}]
res = res.values()
# sudo umount -f /Volumes/wmass ; sudo mkdir -p /Volumes/wmass ; sudo sshfs -o reconnect -o follow_symlinks -o allow_other cyue@lxplus8s10.cern.ch:/home/c/cyue /Volumes/wmass/
# make -j256 ; export OMP_STACKSIZE=512000 ; ./mcfm input_Z_Inclusive.ini
# loginctl enable-linger
# systemd-run --scope --user screen -S Session0
# ctrl+A+D

# General information
nproc = "31" # 31 (Z), 1 (W+), 6(W-)
part = "resaboveNNLO" # resNNLOp bug
# resonlyNNLO, resexpNNLO, resaboveNNLO
# nnlo, resLO, resonlyNNLO, resNLO, resNLOp, resNNLO, resNNLOp, resonlyNNLOp, resonlyN3LO
# most accurate: resonlyNNLO, resNLOp, resNNLO, resonlyN3LO
lhapdfs = ["NNPDF31", "MSHT20"] # NNPDF31 or MSHT20
ewcorrs = ["none", "sudakov", "exact"]
ewcorr = "none"
pgoals = ["0.01", "0.005", "0.001"]
pgoal = "0.01"
name = "_".join([nproc, part, ewcorr, pgoal])

# Select the files to be compared
def sel_files_res(parts_cp):
    files_cp_res_pT = []
    files_cp_res_eta = []
    for part in parts_cp:
        for lhapdf in lhapdfs:
            files_cp_res_pT.append(f"Result MCFM Inclusive/Z_data_{lhapdf}_31_" + part + "_none_0.001_pT.txt")
            files_cp_res_eta.append(f"Result MCFM Inclusive/Z_data_{lhapdf}_31_" + part + "_none_0.001_eta.txt")
    return files_cp_res_pT, files_cp_res_eta
parts_cp = ["resNLOp", "resNNLO"]

# # Get the command to copy the file to /eos/home-c/cyue
# suffix = "_1.0E-4" if ewcorr == "none" and part == "nnlo" or part == "nnlo" else ""
# original_file_pT_res = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_fine.txt"
# original_file_eta_res = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_y34.txt"

# original_files_res = [f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_{i}.txt" for i in range(3, 18+1)]

# copy_commands = [f"cp {file} /eos/home-c/cyue ;" for file in original_files_res]
# copy_commands.extend([f"cp {original_file_pT_res} /eos/home-c/cyue ;", f"cp {original_file_eta_res} /eos/home-c/cyue ;"])
# print("cd Z13TeV ;", " ".join(copy_commands), "cd ..")

# # Get the command to copy the file to /eos/home-c/cyue
# suffix = "_1.0E-4" if ewcorr == "none" and part == "nnlo" or part == "nnlo" else ""
# original_file_pT_res = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}MSHT20nnlo_as118_1.00_1.00{suffix}_Z13TeV_pt34_fine.txt"
# original_file_eta_res = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}MSHT20nnlo_as118_1.00_1.00{suffix}_Z13TeV_y34.txt"

# original_files_res = [f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}MSHT20nnlo_as118_1.00_1.00{suffix}_Z13TeV_pt34_{i}.txt" for i in range(3, 18+1)]

# copy_commands = [f"cp {file} /eos/home-c/cyue ;" for file in original_files_res]
# copy_commands.extend([f"cp {original_file_pT_res} /eos/home-c/cyue ;", f"cp {original_file_eta_res} /eos/home-c/cyue ;"])
# print("cd Z13TeV ;", " ".join(copy_commands), "cd ..")

# # Copy the transverse momentum data and rapidity data to the destination folder with the new names
# shutil.copy("/eos/home-c/cyue/" + original_file_pT_res, "Result MCFM Inclusive/Z_data_NNPDF31_{}_pT.txt".format(name))
# shutil.copy("/eos/home-c/cyue/" + original_file_eta_res, "Result MCFM Inclusive/Z_data_NNPDF31_{}_eta.txt".format(name))
# i = 3
# for file in original_files_res:
#     shutil.copy("/eos/home-c/cyue/" + file, f"Result MCFM Inclusive/Z_data_NNPDF31_{name}_pT_{i}.txt")
#     i += 1
# print(i)

# # Copy the transverse momentum data and rapidity data to the destination folder with the new names
# shutil.copy("/eos/home-c/cyue/" + original_file_pT_res, "Result MCFM Inclusive/Z_data_MSHT20_{}_pT.txt".format(name))
# shutil.copy("/eos/home-c/cyue/" + original_file_eta_res, "Result MCFM Inclusive/Z_data_MSHT20_{}_eta.txt".format(name))
# i = 3
# for file in original_files_res:
#     shutil.copy("/eos/home-c/cyue/" + file, f"Result MCFM Inclusive/Z_data_MSHT20_{name}_pT_{i}.txt")
#     i += 1
# print(i)

# Define the bins
bins_res_pT = np.arange(0, 101, 1)
bins_res_eta = np.append(np.arange(0, 3.75, 0.25), [4, 5])
bin_res_widths_pT = np.diff(bins_res_pT).reshape((-1, 1))
bin_res_widths_eta = np.diff(bins_res_eta)
bin_res_widths = (bin_res_widths_eta * bin_res_widths_pT).flatten()

# Compare transverse momentum plots with DYTurbo resummation
def compare_res_pT(parts_cp, pT_low):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_pT = bins_res_pT
    edges_pT_centres = (edges_pT[:-1] + edges_pT[1:]) / 2
    values_pT = np.reshape(np.sum(res, axis=1), (1, 100)).flatten() / np.diff(edges_pT)
    label_DYTurbo_res = r"DYTurbo resummed, $\mathrm{N}^{3}$LL+NNLO"
    ax1.hist(edges_pT[:-1], bins=edges_pT, weights=values_pT, histtype="step", label=label_DYTurbo_res)
    files_cp_res_pT = check_pgoal(sel_files_res(parts_cp)[0])
    for i, file in enumerate(files_cp_res_pT):
        try:
            lhapdf = file.split("_")[2]
            part = file.split("_")[4]
            name = file[len(f"Result MCFM Inclusive/Z_data_{lhapdf}_"):-len(".txt")]
            label = part_dict[part] + f" ({lhapdf})"
            edges, values, errors = extract(file)
            edges_centres = (edges[:-1] + edges[1:]) / 2
            yerr = errors
            ax1.hist(edges_centres, bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
            ax1.fill_between(edges, np.append(values-yerr, values[-1]-yerr[-1]), np.append(values+yerr, values[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
            ratios = np.append(values/values_pT, values[-1]/values_pT[-1])
            ratios_errors = np.append(yerr/values_pT, yerr[-1]/values_pT[-1])
            ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
            ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        except:
            pass
    ax1.set_xlabel(r"$p_T^Z$ (GeV)")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $p_T^Z$")
    ax1.legend(loc="upper right")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$p_T^Z$ (GeV)")
    ax2.set_ylabel("Pred./DYTurbo resummed")
#     ax2.legend()
    ax2.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
#     ax2.set_ylim(0.8, 1.1)
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo res/sigma vs pT{'' if pT_low is None else pT_low} for DYTurbo resummed {len(parts_cp)} sets Z.png")
    plt.show()

compare_res_pT(parts_cp, None)
compare_res_pT(["resonlyNNLO", "resexpNNLO", "resaboveNNLO"], None)

# Combine three pieces of resNNLO and compare transverse momentum plots
def comb_resNNLO_pT(lhapdf, pT_low):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_pT = bins_res_pT
    edges_pT_centres = (edges_pT[:-1] + edges_pT[1:]) / 2
    edges_pT, values_pT, errors_pT = extract(f"Result MCFM Inclusive/Z_data_{lhapdf}_31_resNNLO_none_0.01_pT.txt")
    ax1.hist(edges_pT[:-1], bins=edges_pT, weights=values_pT, histtype="step", label=f"resNNLO ({lhapdf})", color=colors[0])
    ax1.fill_between(edges_pT, np.append(values_pT-errors_pT, values_pT[-1]-errors_pT[-1]), np.append(values_pT+errors_pT, values_pT[-1]+errors_pT[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_res_pT = check_pgoal(sel_files_res(["resonlyNNLO", "resexpNNLO", "resaboveNNLO"])[0])
    values_sum = np.array([]) 
    errors_sum = np.array([]) 
    for file in files_cp_res_pT:
        if "NNPDF31" in file:
            edges, values, errors = extract(file)
            if len(values_sum) == 0:
                values_sum = values
                errors_sum = errors
            else:
                values_sum += values
                errors_sum += errors
    yerr = errors_sum
    ax1.hist(edges_pT[:-1], bins=edges_pT, weights=values_sum, histtype="step", label=f"sum ({lhapdf})", color=colors[1])
    ax1.fill_between(edges_pT, np.append(values_sum-yerr, values_sum[-1]-yerr[-1]), np.append(values_sum+yerr, values_sum[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[1])
    ratios = np.append(values_sum/values_pT, values_sum[-1]/values_pT[-1])
    ratios_errors = ((errors_sum/values_pT)**2+(values_sum/values_pT**2*errors_pT)**2)**(1/2)
    ratios_errors = np.append(ratios_errors, ratios_errors[-1])
    ax2.step(edges_pT, ratios, where="post", linestyle="-", label="sum", color=colors[1])
    ax2.fill_between(edges_pT, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[1])
    ax1.set_xlabel(r"$p_T^Z$ (GeV)")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $p_T^Z$")
    ax1.legend(loc="upper right")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$p_T^Z$ (GeV)")
    ax2.set_ylabel("sum/resNNLO")
#     ax2.legend()
    ax2.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Inclusive/sigma vs pT{'' if pT_low is None else pT_low} for resNNLO {lhapdf} Z.png")
    plt.show()

comb_resNNLO_pT("NNPDF31", None)
comb_resNNLO_pT("MSHT20", None)

# Compare normalized transverse momentum plots with DYTurbo resummation
def compare_norm_res_pT(parts_cp, pT_low):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_pT = bins_res_pT
    edges_pT_centres = (edges_pT[:-1] + edges_pT[1:]) / 2
    values_pT = np.reshape(np.sum(res, axis=1), (1, 100)).flatten() / np.diff(edges_pT)
    values_pT_norm = values_pT/sum(values_pT)
    label_DYTurbo_res = r"DYTurbo resummed, $\mathrm{N}^{3}$LL+NNLO"
    ax1.hist(edges_pT[:-1], bins=edges_pT, weights=values_pT_norm, histtype="step", label=label_DYTurbo_res)
    files_cp_res_pT = check_pgoal(sel_files_res(parts_cp)[0])
    for i, file in enumerate(files_cp_res_pT):
        try:
            lhapdf = file.split("_")[2]
            part = file.split("_")[4]
            name = file[len(f"Result MCFM Inclusive/Z_data_{lhapdf}_"):-len(".txt")]
            label = part_dict[part] + f" ({lhapdf})"
            edges, values, errors = extract(file)
            values_norm = values/sum(values)
            edges_centres = (edges[:-1] + edges[1:]) / 2
            yerr = (sum(values)-values)/(sum(values))**2*errors  
            ax1.hist(edges[:-1], bins=edges, weights=values_norm, histtype="step", label=label, color=colors[i+1])
            ax1.fill_between(edges, np.append(values_norm-yerr, values_norm[-1]-yerr[-1]), np.append(values_norm+yerr, values_norm[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
            ratios = np.append(values_norm/values_pT_norm, values_norm[-1]/values_pT_norm[-1])
            ratios_errors = np.append(yerr/values_pT_norm, yerr[-1]/values_pT_norm[-1])
            ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
            ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        except:
            pass
    ax1.set_xlabel(r"$p_T^Z$ (GeV)")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $p_T^Z$")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$p_T^Z$ (GeV)")
    ax2.set_ylabel("Pred./DYTurbo resummed")
#     ax2.legend()
    ax2.set_xlim(edges_pT[0], edges_pT[-1] if pT_low is None else pT_low)
#     ax2.set_ylim(0.8, 1.1)
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo res/Normalized sigma vs pT{'' if pT_low is None else pT_low} for DYTurbo resummed {len(parts_cp)} sets Z.png")
    plt.show()

compare_norm_res_pT(parts_cp, None)
compare_norm_res_pT(parts_cp, 20)
compare_norm_res_pT(["resonlyNNLO", "resexpNNLO", "resaboveNNLO"], None)

# Compare rapidity plots with DYTurbo resummation
def compare_res_eta(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_eta = bins_res_eta
    edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
    values_eta = np.sum(res, axis=2).flatten() / np.diff(edges_eta)
    label_DYTurbo_res = r"DYTurbo resummed, $\mathrm{N}^{3}$LL+NNLO"
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta, histtype="step", label=label_DYTurbo_res)
    files_cp_res_eta = check_pgoal(sel_files_res(parts_cp)[1])
    for i, file in enumerate(files_cp_res_eta):
        try:
            lhapdf = file.split("_")[2]
            part = file.split("_")[4]
            name = file[len(f"Result MCFM Inclusive/Z_data_{lhapdf}_"):-len(".txt")]
            label = part_dict[part] + f" ({lhapdf})"
            edges, values, errors = extract(file)
            edges_centres = (edges[:-1] + edges[1:]) / 2
            yerr = errors
            ax1.hist(edges_centres, bins=edges, weights=values, histtype="step", label=label, color=colors[i+1])
            ax1.fill_between(edges, np.append(values-yerr, values[-1]-yerr[-1]), np.append(values+yerr, values[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
            ratios = np.append(values/values_eta, values[-1]/values_eta[-1])
            ratios_errors = np.append(yerr/values_eta, yerr[-1]/values_eta[-1])
            ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
            ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        except:
            pass
    ax1.set_xlabel(r"$|Y^Z|$")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $|Y^Z|$")
    ax1.legend(loc="center left")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_eta[0], edges_eta[-1])
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$|Y^Z|$")
    ax2.set_ylabel("Pred./DYTurbo resummed")
#     ax2.legend()
    ax2.set_xlim(edges_eta[0], edges_eta[-1])
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo res/sigma vs abs(Y) for DYTurbo resummed {len(parts_cp)} sets Z.png")
    plt.show()

compare_res_eta(parts_cp)
compare_res_eta(["resonlyNNLO", "resexpNNLO", "resaboveNNLO"])

# Combine three pieces of resNNLO and compare rapidity plots
def comb_resNNLO_eta(lhapdf):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_eta = bins_res_eta
    edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
    edges_eta, values_eta, errors_eta = extract(f"Result MCFM Inclusive/Z_data_{lhapdf}_31_resNNLO_none_0.01_eta.txt")
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta, histtype="step", label=f"resNNLO ({lhapdf})", color=colors[0])
    ax1.fill_between(edges_eta, np.append(values_eta-errors_eta, values_eta[-1]-errors_eta[-1]), np.append(values_eta+errors_eta, values_eta[-1]+errors_eta[-1]), alpha=0.3, step="post", linewidth=0, color=colors[0])
    files_cp_res_eta = check_pgoal(sel_files_res(["resonlyNNLO", "resexpNNLO", "resaboveNNLO"])[1])
    values_sum = np.array([]) 
    errors_sum = np.array([]) 
    for file in files_cp_res_eta:
        if "NNPDF31" in file:
            edges, values, errors = extract(file)
            if len(values_sum) == 0:
                values_sum = values
                errors_sum = errors
            else:
                values_sum += values
                errors_sum += errors
    yerr = errors_sum
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_sum, histtype="step", label=f"sum ({lhapdf})", color=colors[1])
    ax1.fill_between(edges_eta, np.append(values_sum-yerr, values_sum[-1]-yerr[-1]), np.append(values_sum+yerr, values_sum[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[1])
    ratios = np.append(values_sum/values_eta, values_sum[-1]/values_eta[-1])
    ratios_errors = ((errors_sum/values_eta)**2+(values_sum/values_eta**2*errors_eta)**2)**(1/2)
    ratios_errors = np.append(ratios_errors, ratios_errors[-1])
    ax2.step(edges_eta, ratios, where="post", linestyle="-", label="sum", color=colors[1])
    ax2.fill_between(edges_eta, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[1])
    ax1.set_xlabel(r"$|Y^Z|$")
    ax1.set_ylabel(r"$\sigma$ (pb)")
    ax1.set_title(fr"$\sigma$ vs $|Y^Z|$")
    ax1.legend(loc="upper right")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_eta[0], edges_eta[-1])
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$|Y^Z|$")
    ax2.set_ylabel("sum/resNNLO")
#     ax2.legend()
    ax2.set_xlim(edges_eta[0], edges_eta[-1])
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Inclusive/sigma vs abs(Y) for resNNLO {lhapdf} Z.png")
    plt.show()

comb_resNNLO_eta("NNPDF31")
comb_resNNLO_eta("MSHT20")

# Compare normalized rapidity plots with DYTurbo resummation
def compare_norm_res_eta(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]})
    edges_eta = bins_res_eta
    edges_eta_centres = (edges_eta[:-1] + edges_eta[1:]) / 2
    values_eta = np.sum(res, axis=2).flatten() / np.diff(edges_eta)
    values_eta_norm = values_eta/sum(values_eta)
    label_DYTurbo_res = r"DYTurbo resummed, $\mathrm{N}^{3}$LL+NNLO"
    ax1.hist(edges_eta[:-1], bins=edges_eta, weights=values_eta_norm, histtype="step", label=label_DYTurbo_res)
    files_cp_res_eta = check_pgoal(sel_files_res(parts_cp)[1])
    for i, file in enumerate(files_cp_res_eta):
        try:
            lhapdf = file.split("_")[2]
            part = file.split("_")[4]
            name = file[len(f"Result MCFM Inclusive/Z_data_{lhapdf}_"):-len(".txt")]
            label = part_dict[part] + f" ({lhapdf})"
            edges, values, errors = extract(file)
            values_norm = values/sum(values)
            edges_centres = (edges[:-1] + edges[1:]) / 2
            yerr = (sum(values)-values)/(sum(values))**2*errors
            ax1.hist(edges_centres, bins=edges, weights=values_norm, histtype="step", label=label, color=colors[i+1])
            ax1.fill_between(edges, np.append(values_norm-yerr, values_norm[-1]-yerr[-1]), np.append(values_norm+yerr, values_norm[-1]+yerr[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i+1])
            ratios = np.append(values_eta_norm/values_norm, values_eta_norm[-1]/values_norm[-1])
            ratios_errors = np.append(yerr/values_norm, yerr[-1]/values_norm[-1])
            ax2.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i+1])
            ax2.fill_between(edges, ratios - ratios_errors, ratios + ratios_errors, alpha=0.3, step="post", linewidth=0, color=colors[i+1])
        except:
            pass
    ax1.set_xlabel(r"$|Y^Z|$")
    ax1.set_ylabel(r"Normalized $\sigma$")
    ax1.set_title(r"Normalized $\sigma$ vs $|Y^Z|$")
    ax1.legend(loc="lower left")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(edges_eta[0], edges_eta[-1])
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel(r"$|Y^Z|$")
    ax2.set_ylabel("Pred./DYTurbo resummed")
#     ax2.legend()
    ax2.set_xlim(edges_eta[0], edges_eta[-1])
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo res/Normalized sigma vs abs(Y) for DYTurbo resummed {len(parts_cp)} sets Z.png")
    plt.show()

compare_norm_res_eta(parts_cp)
compare_norm_res_eta(["resonlyNNLO", "resexpNNLO", "resaboveNNLO"])

# Plot variable vs DYTurbo resummation
def plot_vs_res_DYTurbo(parts_cp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]}, sharex=True)
    values_Zmumu = res.T.flatten()
    label_DYTurbo_res = r"DYTurbo resummed, $\mathrm{N}^{3}$LL+NNLO"
    ax1.plot(np.arange(len(values_Zmumu)), values_Zmumu/bin_res_widths, linestyle="-", label=label_DYTurbo_res)
    next(ax2._get_lines.prop_cycler)
    for lhapdf in lhapdfs:
        try:
            for part in parts_cp:
                nproc, ewcorr, pgoal = "31", "none", "0.001"
                label = part_dict[part] + f" ({lhapdf})"
                name = "_".join([nproc, part, ewcorr, pgoal])
                files = []
                values_array = []
                errors_array = []
                for i in range(3, 18+1):
                    files.append(f"Z_data_{lhapdf}_{name}_pT_{i}.txt")
                files = check_pgoal(files)
                for file in files:
                    if not file.startswith("Result MCFM Inclusive/"):
                        file = "Result MCFM Inclusive/" + file
                    edges, values, errors = extract_bin(file)
                    values_array.append(values)
                    errors_array.append(errors)
                values = np.array(values_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
                errors = np.array(errors_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
                ax1.plot(np.arange(len(values)), values/bin_res_widths, linestyle=":", label=f"{label}")
                ratios = values/values_Zmumu
                ax2.plot(np.arange(len(values)), ratios, "-", label=f"{label}")
        except:
            pass
    ax1.set_xlabel("bin")
    ax1.set_ylabel(r"$d\sigma$/bin")
    ax1.set_title(r"$\sigma$ vs bin")
    ax1.legend()
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(0, len(values_Zmumu))
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel("bin")
    ax2.set_ylabel("Pred./DYTurbo resummed")
#     ax2.legend()
    plt.tight_layout()
    plt.savefig(f"Figure DYTurbo res/sigma vs bin for DYTurbo resummed {len(parts_cp)} sets Z.png")
    plt.show()

plot_vs_res_DYTurbo(parts_cp)
plot_vs_res_DYTurbo(["resonlyNNLO", "resexpNNLO", "resaboveNNLO"])

# Combine three pieces of resNNLO and compare bin plots
def comb_resNNLO_bin(lhapdf):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [5, 1]}, sharex=True)
    values_Zmumu = np.array([])
    errors_Zmumu = np.array([])
    values_sum = np.array([])
    errors_sum = np.array([])
    for part in ["resNNLO", "resonlyNNLO", "resexpNNLO", "resaboveNNLO"]:
        nproc, ewcorr, pgoal = "31", "none", "0.001"
        label = part_dict[part] + f" ({lhapdf})"
        name = "_".join([nproc, part, ewcorr, pgoal])
        files = []
        values_array = []
        errors_array = []
        for i in range(3, 18+1):
            files.append(f"Z_data_{lhapdf}_{name}_pT_{i}.txt")
        files = check_pgoal(files)
        for file in files:
            if not file.startswith("Result MCFM Inclusive/"):
                file = "Result MCFM Inclusive/" + file
            edges, values, errors = extract_bin(file)
            values_array.append(values)
            errors_array.append(errors)
        values = np.array(values_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        errors = np.array(errors_array).T.flatten() # Transpose the 2D array and flatten it into a 1D array
        if part == "resNNLO":
            values_Zmumu = values
            errors_Zmumu = errors
        else:
            if len(values_sum) == 0:
                values_sum = values
                errors_sum = errors
            else:
                values_sum += values
                errors_sum += errors
    ax1.plot(np.arange(len(values_Zmumu)), values_Zmumu/bin_res_widths, linestyle=":", label=f"resNNLO ({lhapdf})")
    ax1.plot(np.arange(len(values_sum)), values_sum/bin_res_widths, linestyle=":", label=f"sum ({lhapdf})")
    ax1.legend()
    ax1.set_xlabel("bin")
    ax1.set_ylabel(r"$d\sigma$/bin")
    ax1.set_title(r"$\sigma$ vs bin")
    ax1.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=ax1.transAxes, ha="right", va="top")
    ax1.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=ax1.transAxes, fontweight="bold", va="top", ha="left")
    ax1.text(0.085, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=ax1.transAxes, style="italic", va="top", ha="left")
    ax1.set_xlim(0, len(values_Zmumu))
    next(ax2._get_lines.prop_cycler)
    ratios = values_sum/values_Zmumu
    ax2.plot(np.arange(len(values_sum)), ratios, "-", label=f"sum ({lhapdf})")
    ax2.axhline(1.0, linestyle="--")
    ax2.set_xlabel("bin")
    ax2.set_ylabel("sum/resNNLO")
    ax2.set_ylim(0, 2)
#     ax2.legend()
    plt.tight_layout()
    plt.savefig(f"Figure MCFM Inclusive/sigma vs bin for resNNLO {lhapdf} Z.png")
    plt.show()
    
comb_resNNLO_bin("NNPDF31")
comb_resNNLO_bin("MSHT20")

# Find chi squared value from 2D bins plot, have pT low cut
def chi_sq(part, lhapdf, pT_low):
    edges_pT = bins_res_pT
    edges_eta = bins_res_eta
    if pT_low == None:
        cut = None
    else:
        cut = (len([x for x in edges_pT if x <= pT_low])-1)*(len(edges_eta)-1)
    values_Zmumu = res.T.flatten()[:cut]
    nproc, ewcorr, pgoal = "31", "none", "0.001"
    label = part_dict[part]
    name = "_".join([nproc, part, ewcorr, pgoal])
    files = []
    values_array = []
    errors_array = []
    for i in range(3, 18+1):
        files.append(f"Z_data_{lhapdf}_{name}_pT_{i}.txt")
    files = check_pgoal(files)
    for file in files:
        if not file.startswith("Result MCFM Inclusive/"):
            file = "Result MCFM Inclusive/" + file
        edges, values, errors = extract_bin(file)
        values_array.append(values)
        errors_array.append(errors)
    values = np.array(values_array).T.flatten()[:cut] # Transpose the 2D array and flatten it into a 1D array
    errors = np.array(errors_array).T.flatten()[:cut] # Transpose the 2D array and flatten it into a 1D array
    chi_sq = ((values_Zmumu-values)**2/errors**2).sum()/len(values)
    return chi_sq

# Find the lease chi squared value from 2D bins plot, have pT low cut
def least_chi_sq(parts, lhapdf, pT_low):
    parts = sorted(parts, key=lambda part: chi_sq(part, lhapdf, pT_low), reverse=False)[:int(len(parts))]
    for part in parts:
        print(f"{part} ({lhapdf}) gives a reduced chi_squared of {chi_sq(part, lhapdf, pT_low):.2e}.")

least_chi_sq(parts_2d, "NNPDF31", None)
least_chi_sq(parts_2d, "NNPDF31", 30)
least_chi_sq(parts_2d, "MSHT20", None)
least_chi_sq(parts_2d, "MSHT20", 30)
