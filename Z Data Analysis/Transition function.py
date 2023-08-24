import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
# Set the colour palette to "bright"
sns.set_palette("bright")
folders = ["Figure Transition Function"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Define the transition function
def s(x, l, r, u):
    m = (r+l)/2
    w = (r-l)/2
    return 1/(1+np.exp(np.log((1-u)/u)*(x-m)/w))
xmin = 0.001
xmax = 0.4
u = 0.001
def t(x, xmin, xmax, u):
    if x < xmin:
        return 1
    else:
        return s(x, xmin, xmax, u)/s(xmin, xmin, xmax, u)

# Plot the transition function
xs = np.linspace(0, 1, 10000)
ys = [t(x, xmin, xmax, u) for x in xs]
fig = plt.figure(figsize=(8, 6))
plt.plot(xs, ys)
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$t(\lambda)$")
plt.grid(True)
plt.title("transition function")
plt.savefig("Figure Transition Function/transition function.png")
plt.show()
