# Theoretical-Validation-MCFM-W-Mass
This repository contains files for analyzing MCFM fixed-order calculations of Z and W dileptonic decays.
# Data
The Data folder contains the files related to the analysis of MCFM calculations.
# Data/PDFs
The Data/PDFs subfolder contains codes that study the two PDF sets used in MCFM: NNPDF31_nnlo_as_0118 and MSHT20nnlo_as118.
Additionally, a perturbative toy PDF model is constructed and compared with the PDF sets.
# MCFM-10.3
The MCFM-10.3 folder contains codes that define input configurations and binning of histograms produced in MCFM.
To start the program, first go to Bin
```bash
cd MCFM-10.3/Bin
```
and run
```bash
make -j256 ; export OMP_STACKSIZE=512000 ; ./mcfm input_Z.ini
```
which you should replace input_Z.ini with the desired file.
