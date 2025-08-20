import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# Update rcParams for larger figures and fonts
plt.rcParams.update({
    'figure.figsize': (18, 6),  # Increase default figure size
    'font.size': 18,           # Increase font size for all text
    'axes.titlesize': 16,      # Increase title font size
    'axes.labelsize': 16,      # Increase x and y label font size
    'xtick.labelsize': 16,     # Increase x-tick label font size
    'ytick.labelsize': 16,     # Increase y-tick label font size
    'legend.fontsize': 16      # Increase legend font size
})

# Load the data
data = pd.DataFrame({
    "model": ["DeiT", "Swin", "ViT", "EACE-ViTs"],
    "acc": [0.03568120160740312, 0.044139579394640846, 0.04709951377170328, 0.025609845714702425],
    "cov_thr": [0.06646381174447441, 0.08038775626937654, 0.06807681073973594, 0.06303780954394561],
    "cov_raps": [0.05826715823167508, 0.06292295161152384, 0.059129813770577765, 0.041666666666666664],
    "cov_aps": [0.04715068110213319, 0.05329606626208304, 0.11032709504575278, 0.06303780954394565],
    "size_thr": [0.11580785091406078, 0.0949384547229687, 0.24689428936987742, 0.07482618748361658],
    "size_raps": [0.34777733404302535, 0.06292295161152385, 0.7718024438583226, 0.08895939044388824],
    "size_aps": [0.5005253798402209, 0.21309376015567413, 2.1694915701240935, 0.11669973076444443]
})

# Error margin (example, replace with actual errors if available)
yerr = [0.001] * 4

# Create a 1x3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("Standard Deviation Metrics for Each Model", fontsize=18)

# Plot standard deviation of accuracy
sns.scatterplot(x="model", y="acc", data=data, s=100, color="dodgerblue", ax=axes[0], label="Accuracy")
axes[0].errorbar(data["model"], data["acc"], yerr=yerr, fmt='o', color="dodgerblue", capsize=4)
axes[0].set_ylabel("Std Dev of Accuracy")
axes[0].set_title("(a) Accuracy", fontsize=16)

# Plot standard deviation of coverage (RAPS, APS)
sns.scatterplot(x="model", y="cov_raps", data=data, s=100, color="mediumseagreen", ax=axes[1], label="RAPS Coverage")
sns.scatterplot(x="model", y="cov_aps", data=data, s=100, color="skyblue", ax=axes[1], label="APS Coverage")
axes[1].errorbar(data["model"], data["cov_raps"], yerr=yerr, fmt='o', color="mediumseagreen", capsize=4)
axes[1].errorbar(data["model"], data["cov_aps"], yerr=yerr, fmt='o', color="skyblue", capsize=4)
axes[1].set_ylabel("Std Dev of Coverage")
axes[1].set_title("(b) Coverage (RAPS, APS)", fontsize=16)
axes[1].legend()

# Plot standard deviation of prediction set size (RAPS, APS)
sns.scatterplot(x="model", y="size_raps", data=data, s=100, color="mediumseagreen", ax=axes[2], label="RAPS Set Size")
sns.scatterplot(x="model", y="size_aps", data=data, s=100, color="skyblue", ax=axes[2], label="APS Set Size")
axes[2].errorbar(data["model"], data["size_raps"], yerr=yerr, fmt='o', color="mediumseagreen", capsize=4)
axes[2].errorbar(data["model"], data["size_aps"], yerr=yerr, fmt='o', color="skyblue", capsize=4)
axes[2].set_ylabel("Std Dev of Prediction Set Size")
axes[2].set_title("(c) Prediction Set Size (RAPS, APS)", fontsize=16)
axes[2].legend()

# Adjust layout to avoid overlapping titles and labels
plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Leaves space for the main title
plt.savefig('STD.png', dpi=300)  # High-resolution save
plt.show()
