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
if not os.path.exists("Figure resNNLOp Bug"):
    os.makedirs("Figure resNNLOp Bug")
# sudo umount -f /Volumes/wmass ; sudo mkdir -p /Volumes/wmass ; sudo sshfs -o reconnect -o follow_symlinks -o allow_other cyue@lxplus8s10.cern.ch:/home/c/cyue /Volumes/wmass/
# make -j256 ; export OMP_STACKSIZE=512000 ; ./mcfm input_Z.ini
# loginctl enable-linger
# systemd-run --scope --user screen -S Session0

# General information
nproc = "31" # 31 (Z), 1 (W+), 6(W-)
part = "resNNLOp" # resNNLOp bugx
# most accurate: resonlyNNLO, resNLOp, resNNLO, resonlyN3LO
ewcorrs = ["none", "sudakov", "exact"]
ewcorr = "exact"
pgoals = ["0.01", "0.005", "0.001"]
pgoal = "0.01"
version = "10.3" # 10.2.2, 10.3
name = "_".join([nproc, part, ewcorr, pgoal, version])
fixed = True
if fixed == True:
    name = name+"_fixed"

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
original_file_pT = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}_NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_fine.txt"
original_file_eta = f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}_NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_y34.txt"

original_files = [f"Z_only_{part}_{ewcorr if ewcorr != 'none' else ''}_NNPDF31_nnlo_as_0118_1.00_1.00{suffix}_Z13TeV_pt34_{i}.txt" for i in range(3, 12+1)]

copy_commands = [f"cp {file} /eos/home-c/cyue ;" for file in original_files]
copy_commands.extend([f"cp {original_file_pT} /eos/home-c/cyue ;", f"cp {original_file_eta} /eos/home-c/cyue ;"])
print("cd Z13TeV ;", " ".join(copy_commands), "cd ..")

# # Copy the transverse momentum data and rapidity data to the destination folder with the new names
# shutil.copy("/eos/home-c/cyue/" + original_file_pT, "Result MCFM/Z_data_{}_pT.txt".format(name))
# shutil.copy("/eos/home-c/cyue/" + original_file_eta, "Result MCFM/Z_data_{}_eta.txt".format(name))

# Define the colours
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Extract data from text file
def extract(file):
    # Open the text file
    with open(file, "r") as f:
        # Skip the first 5 rows
        for i in range(5):
            f.readline()
        # Extract the data
        data = [line.split() for line in f.readlines()]
        if file.endswith("pT.txt"):
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
    version = file.split("_")[6]
    if (version == "10.2.2" or version == "10.3"):
        plot(file)

# Select files to be compared
def sel_files(parts_cp):
    files_cp_pT = []
    files_cp_eta = []
    for part in parts_cp:
        if part == "resonlyN3LO":
            files_cp_pT.append("Result MCFM/Z_data_31_" + part + "_exact_0.01" + "_10.3" + "_pT.txt")
            files_cp_eta.append("Result MCFM/Z_data_31_" + part + "_exact_0.01" + "_10.3" + "_eta.txt")
        else:
            for version in ["10.2.2", "10.3"]:
                files_cp_pT.append(f"Result MCFM/Z_data_31_{part}_exact_0.01_{version}_pT.txt")
                files_cp_eta.append(f"Result MCFM/Z_data_31_{part}_exact_0.01_{version}_eta.txt")
    files_cp_pT.append("Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_fixed_pT.txt")
    files_cp_eta.append("Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_fixed_eta.txt")
    return files_cp_pT, files_cp_eta
parts_cp = ["resonlyNNLO", "resNLOp", "resNNLO", "resNNLOp", "resonlyN3LO"] # resNNLOp bug
# parts_cp = ["resonlyNNLO", "resNLOp", "resNNLO", "resonlyN3LO"]
# parts_cp = ["resNNLOp"]

# Compare transverse momentum plots
def compare_pT(parts_cp):
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    for i, file in enumerate(sel_files(parts_cp)[0]):
        part = file.split("_")[3]
        version = file.split('_')[6]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        if file == "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_fixed_pT.txt":
            label = f"{part_dict[part]} ({version} fixed)"
        else:
            label = f"{part_dict[part]} ({version})"
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        plt.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i])
        plt.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i])
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel(r"$\sigma$ (pb)")
    plt.title(fr"$\sigma$ vs $p_T^Z$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.xlim(edges[0], edges[-1])
    plt.savefig(f"Figure resNNLOp Bug/sigma vs pT for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()

compare_pT(parts_cp)

# Compare rapidity plots
def compare_eta(parts_cp):
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    for i, file in enumerate(sel_files(parts_cp)[1]):
        part = file.split("_")[3]
        version = file.split('_')[6]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        if file == "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_fixed_eta.txt":
            label = f"{part_dict[part]} ({version} fixed)"
        else:
            label = f"{part_dict[part]} ({version})"
        edges, values, errors = extract(file)
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = errors
        plt.hist(edges[:-1], bins=edges, weights=values, histtype="step", label=label, color=colors[i])
        plt.fill_between(edges, np.append(values-errors, values[-1]-errors[-1]), np.append(values+errors, values[-1]+errors[-1]), alpha=0.3, step="post", linewidth=0, color=colors[i])
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel(r"$\sigma$ (pb)")
    plt.title(fr"$\sigma$ vs $|Y^Z|$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.xlim(edges[0], edges[-1])
    plt.savefig(f"Figure resNNLOp Bug/sigma vs abs(Y) for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()

compare_eta(parts_cp)

# Compare transverse momentum plots with resNNLOp 10.3
def compare_ratio_pT_low(parts_cp):
    pT_low = 30
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    file = "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_pT.txt"
    edges, values, errors = extract(file)
    edges = edges[edges <= pT_low]
    values_nom = values[:len(edges)-1]
    errors_nom = errors[:len(edges)-1]
    for i, file in enumerate(sel_files(parts_cp)[0]):
        if file == "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_pT.txt":
            continue # skip this iteration
        else:
            part = file.split("_")[3]
            version = file.split('_')[6]
            name = file[len("Result MCFM/Z_data_"):-len(".txt")]
            if file == "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_fixed_pT.txt":
                label = f"{part_dict[part]} ({version} fixed)"
            else:
                label = f"{part_dict[part]} ({version})"
            edges, values, errors = extract(file)
            edges = edges[edges <= pT_low]
            values = values[:len(edges)-1]
            errors = errors[:len(edges)-1]
            edges_centres = (edges[:-1] + edges[1:]) / 2
            ratios = np.append(values_nom/values, values_nom[-1]/values[-1])
            yerr = np.sqrt((errors_nom/values)**2+(values_nom*errors/values**2)**2)
            yerr = np.append(yerr, yerr[-1])
            plt.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i])
            plt.fill_between(edges, ratios - yerr, ratios + yerr, alpha=0.3, step="post", linewidth=0, color=colors[i])
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel("resNNLOp Bug 10.3/Pred.")
    plt.title(r"resNNLOp 10.3/Pred. vs $p_T^Z$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.xlim(edges[0], edges[-1])
    plt.savefig(f"Figure resNNLOp Bug/ratio vs pT{pT_low} for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()

compare_ratio_pT_low(parts_cp)

# Compare rapidity plots with resNNLOp 10.3
def compare_ratio_eta(parts_cp):
    eta_high = 2
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    file = "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_eta.txt"
    edges, values, errors = extract(file)
    edges = edges[edges <= eta_high]
    values_nom = values[:len(edges)-1]
    errors_nom = errors[:len(edges)-1]
    for i, file in enumerate(sel_files(parts_cp)[1]):
        if file == "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_eta.txt":
            continue # skip this iteration
        else:
            part = file.split("_")[3]
            version = file.split('_')[6]
            name = file[len("Result MCFM/Z_data_"):-len(".txt")]
            if file == "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_fixed_eta.txt":
                label = f"{part_dict[part]} ({version} fixed)"
            else:
                label = f"{part_dict[part]} ({version})"
            edges, values, errors = extract(file)
            edges = edges[edges <= eta_high]
            values = values[:len(edges)-1]
            errors = errors[:len(edges)-1]
            edges_centres = (edges[:-1] + edges[1:]) / 2
            ratios = np.append(values_nom/values, values_nom[-1]/values[-1])
            yerr = np.sqrt((errors_nom/values)**2+(values_nom*errors/values**2)**2)
            yerr = np.append(yerr, yerr[-1])
            plt.step(edges, ratios, where="post", linestyle="-", label=label, color=colors[i])
            plt.fill_between(edges, ratios - yerr, ratios + yerr, alpha=0.3, step="post", linewidth=0, color=colors[i])
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel("resNNLOp Bug 10.3/Pred.")
    plt.title(r"resNNLOp 10.3/Pred. vs $|Y^Z|$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.xlim(edges[0], edges[-1])
    plt.savefig(f"Figure resNNLOp Bug/ratio vs abs(Y) for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()
    
compare_ratio_eta(parts_cp)

# Compare normalized rapidity plots with pseudo data
def compare_norm_eta(parts_cp):
    eta_high = 2
    nproc, ewcorr, pgoal = "31", "exact", "0.01"
    fig = plt.figure(figsize=(8, 6))
    for i, file in enumerate(sel_files(parts_cp)[1]):
        part = file.split("_")[3]
        version = file.split('_')[6]
        name = file[len("Result MCFM/Z_data_"):-len(".txt")]
        if file == "Result MCFM/Z_data_31_resNNLOp_exact_0.01_10.3_fixed_eta.txt":
            label = f"{part_dict[part]} ({version} fixed)"
        else:
            label = f"{part_dict[part]} ({version})"
        edges, values, errors = extract(file)
        edges = edges[edges <= eta_high]
        values = values[:len(edges)-1]
        errors = errors[:len(edges)-1]
        values_norm = values/sum(values)
        values_norm = np.append(values_norm, values_norm[-1])
        edges_centres = (edges[:-1] + edges[1:]) / 2
        yerr = (sum(values)-values)/(sum(values))**2*errors
        yerr = np.append(yerr, yerr[-1])
        plt.step(edges, values_norm, where="post", linestyle="-", label=label, color=colors[i])
        plt.fill_between(edges, values_norm - yerr, values_norm + yerr, alpha=0.3, step="post", linewidth=0, color=colors[i])
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel(r"Normalized $\sigma$")
    plt.title(r"Normalized $\sigma$ vs $|Y^Z|$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.xlim(edges[0], edges[-1])
    plt.savefig(f"Figure resNNLOp Bug/Normalized sigma vs abs(Y) for {nproc}_{ewcorr}_{pgoal} {len(parts_cp)} sets.png")
    plt.show()

compare_norm_eta(parts_cp)
