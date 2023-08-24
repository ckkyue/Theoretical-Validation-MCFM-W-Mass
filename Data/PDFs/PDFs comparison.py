import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from scipy import integrate
from scipy.stats import ks_2samp
from scipy.optimize import curve_fit
folders = ["Figure PDFs"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Define the colours
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Toy model of PDFs
q = 100
q0 = np.sqrt(5)
Lambda = 0.2
nf = 4
b = (33-2*nf)/(12*np.pi)
qlog = True
if qlog:
    xs = [x for x in np.logspace(-5, 0, 20000)]
else:
    xs = [x for x in np.linspace(0, 1, 20000)]
    
a1, a2, a3 = 0.2341, 1.4409, 1.4165

def w(x):
    return np.log(1/x)

def zeta(q):
    return np.log(np.log(q**2/Lambda**2))

def K(q):
    return a1*(np.exp(zeta(q)-zeta(q0))-a2)*np.exp(-a3*np.sqrt(zeta(q)-zeta(q0)))

def xg(x, q):
    return K(q)*np.exp(np.sqrt(12/(b*np.pi)*w(x)*(zeta(q)-zeta(q0))))

def xf(x, q):
    return np.sqrt((zeta(q)-zeta(q0))/(108*b*np.pi*w(x)))*xg(x, q)

# Plot the toy model
xg_arr = [xg(x, q) for x in xs]
xf_arr = [xf(x, q) for x in xs]
plt.plot(xs, xg_arr, label="toy gluons")
plt.plot(xs, xf_arr, label="toy quarks")
plt.xlim(0, 0.2)
plt.ylim(0, 3)
plt.xlabel(r"$x$")
plt.ylabel(r"$xf(x)$")
plt.title(fr"Toy model PDFs at $Q={q}$GeV")
plt.legend()
plt.savefig(f"Figure PDFs/Toy model PDFs at small x with Q={q}GeV.png")
plt.show()

# Extract results from two PDF sets
lhapdf.setVerbosity(0)
pdf1 = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)
pdf2 = lhapdf.mkPDF("MSHT20nnlo_as118", 0)
pdf_dict = {pdf1: "NNPDF31", pdf2: "MSHT20"}
q = 100
qlog = True
if qlog:
    xs = [x for x in np.logspace(-5, 0, 20000)]
else:
    xs = [x for x in np.linspace(0, 1, 20000)]

res1 = np.empty([len(xs), 6])
res2 = np.empty([len(xs), 6])

pdf_res = [(pdf1, res1), (pdf2, res2)]

for pdf, res in pdf_res:
    for ix, x in enumerate(xs):
        fac = 1
        res[ix, 0] = x
        res[ix, 1] = q
        res[ix, 2] = fac * (pdf.xfxQ(2, x, q) - pdf.xfxQ(-2, x, q))  # valence up-quark
        res[ix, 3] = fac * (pdf.xfxQ(1, x, q) - pdf.xfxQ(-1, x, q))  # valence down-quark
        res[ix, 4] = fac * (pdf.xfxQ(0, x, q))  # gluon (or "21")
        res[ix, 5] = fac * (2 * pdf.xfxQ(-2, x, q)+ 2 * pdf.xfxQ(-1, x, q)
                + pdf.xfxQ(3, x, q) + pdf.xfxQ(-3, x, q)
                + pdf.xfxQ(4, x, q) + pdf.xfxQ(-4, x, q)
                + pdf.xfxQ(5, x, q) + pdf.xfxQ(-5, x, q)) # sum over all sea quarks

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [5, 1]}, sharex=True)
ax1.set_xscale("log")
ax1.set_ylim(0, 1)
ax2.set_xscale("log")
ax2.set_ylim(0.5, 1.5)

# Plot the PDFs
labels = ["valence up", "valence down", "gluons", "sea quarks"]
for pdf, res in pdf_res:
    for i, label in enumerate(labels, start=2):
        ax1.plot(res[:, 0], res[:, i], label=f"{label} ({pdf_dict[pdf]})")
# xg_arr = [xg(x, q) for x in xs]
# xf_arr = [xf(x, q) for x in xs]
# ax1.plot(xs, xg_arr, label="toy gluons")
# ax1.plot(xs, xf_arr, label="toy quarks")
ax1.set_title(fr"PDFs at $Q={q}$GeV")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$xf(x)$")
ax1.legend()

# Compare the two PDFs
for i, label in enumerate(labels, start=2):
    ax2.plot(res1[:, 0], res1[:, i]/res2[:, i], label=f"{label}")
ax2.set_title(r"NNPDF31/MSHT20 vs $x$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel("NNPDF31/MSHT20")
ax2.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig("Figure PDFs/PDFs.png", bbox_inches="tight")
plt.show()

# Perform the KS test
ks_res = []
for i, label in enumerate(labels, start=2):
    ks_stat, p_value = ks_2samp(res1[:, i], res2[:, i], alternative="two-sided")
    ks_res.append((ks_stat, p_value))
    print(f"{label}: ks_stat = {ks_stat:.4e}, p_value = {p_value:.4e}")

# Compare toy model with NNPDF31
lhapdf.setVerbosity(0)
pdf1 = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)
pdf_dict = {pdf1: "NNPDF31", pdf2: "MSHT20"}
q = 100
qlog = True
if qlog:
    xs = [x for x in np.logspace(-5, 0, 20000)]
else:
    xs = [x for x in np.linspace(0, 1, 20000)]

res1 = np.empty([len(xs), 6])
res2 = np.empty([len(xs), 6])

pdf_res = [(pdf1, res1)]

for pdf, res in pdf_res:
    for ix, x in enumerate(xs):
        fac = 1
        res[ix, 0] = x
        res[ix, 1] = q
        res[ix, 2] = fac * (pdf.xfxQ(2, x, q) - pdf.xfxQ(-2, x, q))  # valence up-quark
        res[ix, 3] = fac * (pdf.xfxQ(1, x, q) - pdf.xfxQ(-1, x, q))  # valence down-quark
        res[ix, 4] = fac * (pdf.xfxQ(0, x, q))  # gluon (or "21")
        res[ix, 5] = fac * 0.05 * (2 * pdf.xfxQ(-2, x, q)+ 2 * pdf.xfxQ(-1, x, q)
                + pdf.xfxQ(3, x, q) + pdf.xfxQ(-3, x, q)
                + pdf.xfxQ(4, x, q) + pdf.xfxQ(-4, x, q)
                + pdf.xfxQ(5, x, q) + pdf.xfxQ(-5, x, q)) # sum over all sea quarks

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [5, 1]}, sharex=True)
ax1.set_xscale("log")
ax2.set_xscale("log")
ax2.set_ylim(0, 2)

# Plot the PDFs
labels = ["valence up", "valence down", "gluons", "sea quarks"]
for pdf, res in pdf_res:
    for i, label in enumerate(labels, start=2):
        ax1.plot(res[:, 0], res[:, i], label=f"{label} ({pdf_dict[pdf]})")
xg_arr = [xg(x, q) for x in xs]
xf_arr = [xf(x, q) for x in xs]
ax1.plot(xs, xg_arr, label="toy gluons")
ax1.plot(xs, xf_arr, label="toy quarks")
ax1.set_title(fr"Comparison of toy model and NNPDF31 at $Q={q}$GeV")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$xf(x)$")
ax1.legend()

# Compare
ax2.axhline(1.0, linestyle="--", color="black")
ax2.plot(res1[:, 0], xg_arr/res1[:, 4], label="toy gluons", color=colors[4])
ax2.plot(res1[:, 0], xf_arr/res1[:, 5], label="toy quarks", color=colors[5])
ax2.set_title(r"toy model/NNPDF31 vs $x$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel("toy model/NNPDF31")
ax2.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig("Figure PDFs/Compare toy model and NNPDF31.png", bbox_inches="tight")
plt.show()
print(xg_arr/res1[:, 5])

# Define the function you want to fit
def func(x, a1, a2, a3):
    K = a1*(np.exp(zeta(q)-zeta(q0))-a2)*np.exp(-a3*np.sqrt(zeta(q)-zeta(q0)))
    xg = K*np.exp(np.sqrt(12/(b*np.pi)*w(x)*(zeta(q)-zeta(q0))))
    xf = np.sqrt((zeta(q)-zeta(q0))/(108*b*np.pi*w(x)))*xg
    return xg

# Generate some sample data
xs = [x for x in np.logspace(-5, 0, 20000)]
ys = res1[:, 4]

# Fit the function to the data
params, params_covariance = curve_fit(func, xs, ys)

# Extract the fitted parameters
a1, a2, a3 = [round(param, 4) for param in params]

# Print the fitted parameters
print("Fitted parameters:")
print("a1 =", a1)
print("a2 =", a2)
print("a3 =", a3)

# Find the quantum number
res1 = []
res2 = []
pdf_res = [(pdf1, res1), (pdf2, res2)]
for pdf, res in pdf_res:
    for id, label in [(1, "d"), (2, "u"), (3, "s"), (4, "c"), (5, "b")]:
        int_res = scipy.integrate.quad(lambda x: (pdf.xfxQ(id, x, q) - pdf.xfxQ(-id, x, q)) / x, 1e-6, 1, limit=100, epsrel=1e-3)
        res.append((label, "{:.4e}".format(int_res[0])))
print("res1: ", res1)
print("res2: ", res2)

# Find the momentum fraction
res1 = []
res2 = []
int_sum1 = [0, 0]
int_sum2 = [0, 0]
pdf_res_int = [(pdf1, res1, int_sum1), (pdf2, res2, int_sum2)]
for pdf, res, int_sum in pdf_res_int:
    for id, label in [(-5, "bb"), (-4, "cb"), (-3, "sb"), (-2, "ub"), (-1, "db"), (0, "g"), (1, "d"), (2, "u"), (3, "s"), (4, "c"), (5, "b")]:
        int_res = scipy.integrate.quad(lambda x: pdf.xfxQ(id, x, q), 1e-6, 1, limit=100, epsrel=1e-3)
        int_sum[0] += int_res[0]
        int_sum[1] += int_res[1]**2
        res.append((label, "{:.1f}%".format(int_res[0] * 100)))
    res.append(("SUM", "{:.0f}%".format(int_sum[0] * 100)))
print("res1: ", res1)
print("res2: ", res2)
