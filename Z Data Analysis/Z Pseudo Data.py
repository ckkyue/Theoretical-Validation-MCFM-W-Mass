import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import uproot
datafile = "/eos/home-c/cyue/unfolded_data.root"
data = uproot.open(datafile)
data_Zmumu = data["data_Zmumu"].to_hist().values()
if not os.path.exists("Figure Pseudo Data"):
    os.makedirs("Figure Pseudo Data")
# cov_Zmumu = data["covariance_matrix_Zmumu;1"]
# cov_Zmumu.to_hist().axes.edges
# cov_Zmumu.to_hist().values()

# Get pseudo data
filename = "/eos/home-c/cyue/ZMassDileptonCombineInput_xnorm.root"
file = uproot.open(filename)
variables = []
nominal = "Zmumu/xnorm_Zmumu_inclusive;1"
for key in file.keys():
    if key == nominal:
        pass
    else:
        try:
            if file[key].classname == "TH2D":
                variables.append(key)
        except Exception as e:
            pass
print(f"There are {len(variables)} variables in total.")

# Define the label of a variable
def label(variable):
    if variable == nominal:
        var_label = "nominal"
    else:
        var_label = re.sub(re.compile(r"Zmumu/xnorm_Zmumu_|_inclusive;1"), "", variable)
    return var_label

# Luminosity
lum = 16.8*1000

# # Convert data to 2D histogram
def h2d(variable):
    return file[variable].to_hist()
# Project the 2D histogram to x axis
def projectx(variable):
    fig = plt.figure(figsize=(8, 6))
    hist = h2d(variable).project("xaxis")
    values, edges = hist.to_numpy()
    values = values/(edges[1:]-edges[:-1])/lum
    var_label = label(variable)
    plt.hist(edges[:-1], bins=edges, weights=values, histtype="step", edgecolor="black", label=var_label)
    plt.xlabel(r"$p_T^Z$ (GeV)")
    plt.ylabel(r"$\sigma$ (pb)")
    plt.title(fr"$\sigma$ vs $p_T^Z$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.savefig(f"Figure Pseudo Data/sigma vs pT for {var_label}.png")
    plt.show()
# Project the 2D histogram to y axis
def projecty(variable):
    fig = plt.figure(figsize=(8, 6))
    hist = h2d(variable).project("yaxis")
    values, edges = hist.to_numpy()
    values = values/(edges[1:]-edges[:-1])/lum
    var_label = label(variable)
    plt.hist(edges[:-1], bins=edges, weights=values, histtype="step", edgecolor="black", label=var_label)
    plt.xlabel(r"$|Y^Z|$")
    plt.ylabel(r"$\sigma$ (pb)")
    plt.title(fr"$\sigma$ vs $|Y^Z|$")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.savefig(f"Figure Pseudo Data/sigma vs rapidity for {var_label}.png")
    plt.show()

# Define the bins
bins_pT, bins_eta = h2d(nominal).axes.edges
bins_pT = bins_pT.flatten()
bins_eta = bins_eta.flatten()
bin_widths_pT = np.diff(bins_pT).reshape((-1, 1))
bin_widths_eta = np.diff(bins_eta)
bin_widths = (bin_widths_eta * bin_widths_pT).flatten()

projectx("Zmumu/xnorm_Zmumu_inclusive;1")
projecty("Zmumu/xnorm_Zmumu_inclusive;1")

# Flatten the histogram to a 1D array
def flatten(variable):
    return h2d(variable).values().flatten()/bin_widths/lum

# Get pseudo data
data_Zmumu = flatten(nominal)

# Plot variable vs data
def plot_vs_data_Zmumu(variable):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(data_Zmumu)), data_Zmumu, label=r"$Z\to\mu\mu$")
    fv = flatten(variable)
    var_label = label(variable)
    plt.plot(np.arange(len(fv)), fv, color="red", linestyle=":", label=f"{var_label}")
    plt.xlabel("bin")
    plt.ylabel(r"$d\sigma$/bin")
    plt.title(r"$\sigma$ vs bin")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.savefig(f"Figure Pseudo Data/sigma vs bin for {var_label}.png")
    plt.show()

# Plot the ratio of variable vs data
def plot_ratio_data_Zmumu(variable):
    fig = plt.figure(figsize=(8, 6))
    fv = flatten(variable)
    var_label = label(variable)
    plt.plot(np.arange(len(fv)), fv/data_Zmumu, linestyle=":", label=var_label)
    plt.xlabel("bin")
    plt.ylabel("variation/nominal")
    plt.title(r"variation/nominal vs bin")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.savefig(f"Figure Pseudo Data/ratio vs bin for {var_label}.png")
    plt.show()

# Find maximum deviation
def max_deviation(variable):
    fv = flatten(variable)
    deviation_array = (abs((data_Zmumu/sum(data_Zmumu))/(fv/sum(fv))-1))
    return max(deviation_array)

# Find the topmost maximum deviations
def top_max_deviation(number):
    return sorted(variables, key=lambda var: max_deviation(var), reverse=True)[:int(number)]

# Find the least maximum deviations
def top_min_deviation(number):
    return sorted(variables, key=lambda var: max_deviation(var), reverse=False)[:int(number)]

for variable in top_max_deviation(10):
    print(f"{label(variable)} gives a maximum deviation of {max_deviation(variable):.2e}.")
    projectx(variable)
    projecty(variable)
    plot_vs_data_Zmumu(variable)
    plot_ratio_data_Zmumu(variable)

for variable in top_min_deviation(10):
    print(f"{label(variable)} gives a maximum deviation of {max_deviation(variable):.2e}.")
    projectx(variable)
    projecty(variable)
    plot_vs_data_Zmumu(variable)
    plot_ratio_data_Zmumu(variable)

fig = plt.figure(figsize=(8, 6))
for variable in [nominal, "Zmumu/xnorm_Zmumu_pdfAlphaSDown_inclusive;1", "Zmumu/xnorm_Zmumu_pdfAlphaSUp_inclusive;1"]:
    fv = flatten(variable)
    var_label = label(variable)
    plt.plot(np.arange(len(fv)), fv/data_Zmumu, linestyle=":", label=var_label)
    plt.ylabel("variation/nominal")
    plt.title(r"variation/nominal vs bin")
    plt.legend()
    plt.text(1, 1.05, r"16.8 fb$^{-1}$", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, ha="right", va="top")
    plt.text(0, 1.04, "CMS", fontname="Arial", fontsize=16, transform=plt.gca().transAxes, fontweight="bold", va="top", ha="left")
    plt.text(0.095, 1.04, "Preliminary", fontname="Arial", fontsize=14, transform=plt.gca().transAxes, style="italic", va="top", ha="left")
    plt.savefig("Figure Pseudo Data/ratio vs bin for pdfAlphaS.png")
plt.show()

