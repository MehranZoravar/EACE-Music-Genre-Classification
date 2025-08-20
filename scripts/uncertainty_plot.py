import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from src import conformal_prediction as cp
import matplotlib
matplotlib.use('TkAgg')

# Update rcParams for larger figures and fonts
plt.rcParams.update({
    'figure.figsize': (12, 8),  # Increase figure size
    'font.size': 18,           # Increase font size for all text
    'axes.titlesize': 16,      # Increase title font size
    'axes.labelsize': 16,      # Increase x and y label font size
    'xtick.labelsize': 16,     # Increase x-tick label font size
    'ytick.labelsize': 16,     # Increase y-tick label font size
    'legend.fontsize': 16,     # Increase legend font size
    'lines.markersize': 8,     # Increase marker size
    'lines.linewidth': 2       # Increase line width
})

# Function to compute entropy for each softmax output
def compute_entropy(softmax_output):
    epsilon = 1e-12  # To avoid log(0)
    entropy = -np.sum(softmax_output * np.log(softmax_output + epsilon), axis=1)
    return entropy

# Function to adjust the threshold dynamically based on entropy
def adjust_threshold_based_on_entropy(base_tau, entropy, scaling_factor=0.2, num_classes=7):
    max_entropy = np.log(num_classes)
    normalized_entropy = entropy / max_entropy
    adjusted_tau = base_tau + scaling_factor * normalized_entropy
    return adjusted_tau

# Function to calculate uncertainty based on prediction set size, excluding set size 0
def calculate_uncertainty(set_sizes, total_classes=7):
    return [size / total_classes for size in set_sizes if size > 0]

# Function to gather prediction set sizes separately for correct and incorrect predictions
def gather_set_sizes_by_correctness(conf_set, true_labels):
    correct_set_sizes = []
    incorrect_set_sizes = []

    for i, prediction in enumerate(conf_set):
        set_size = np.sum(prediction)
        if set_size > 0:  # Exclude set size 0
            if prediction[true_labels[i]] == 1:  # Correct prediction
                correct_set_sizes.append(set_size)
            else:  # Incorrect prediction
                incorrect_set_sizes.append(set_size)

    return correct_set_sizes, incorrect_set_sizes

# Main function
def main(root_path: str, val_path: str, alpha: float, n_trials: int = 1):
    root_path = Path(root_path)
    val_path = Path(val_path)
    models = ['ViT', 'Swin', 'DeiT', 'EACE_ViTs']
    color_map = plt.get_cmap("tab10")  # Use color map to assign colors to each model

    # Dictionaries to store uncertainty values for all models for RAPS and APS
    uncertainty_data_raps_correct = defaultdict(list)
    uncertainty_data_raps_incorrect = defaultdict(list)
    uncertainty_data_aps_correct = defaultdict(list)
    uncertainty_data_aps_incorrect = defaultdict(list)

    for trial in tqdm(range(n_trials), desc="Processing Trials"):
        print(f'Trial: {trial}')
        for model_idx, model in enumerate(models):
            npz_path = root_path / f'{model}_model' / f'{model}_output_test.npz'
            data = np.load(npz_path)
            cal_smx = data['smx']
            cal_labels = data['labels'].astype(int)

            npz_path_2 = val_path / f'{model}_model' / f'{model}_output_cal.npz'
            val_data = np.load(npz_path_2)
            val_smx = val_data['smx']
            val_labels = val_data['labels'].astype(int)

            # Calibrate thresholds for RAPS and APS
            tau_raps = cp.calibrate_raps_adjusted(cal_smx, cal_labels, alpha=alpha, rng=True, k_reg=2, lambda_reg=0.1)
            tau_aps = cp.calibrate_raps_adjusted(cal_smx, cal_labels, alpha=alpha, rng=True)

            # Adjust thresholds based on entropy
            val_entropy = compute_entropy(val_smx)
            adjusted_tau_raps = adjust_threshold_based_on_entropy(tau_raps, val_entropy, scaling_factor=0.1)
            adjusted_tau_aps = adjust_threshold_based_on_entropy(tau_aps, val_entropy, scaling_factor=0.1)

            # Predict confidence sets using adjusted taus
            conf_set_raps = cp.predict_raps(val_smx, adjusted_tau_raps, rng=True, k_reg=2, lambda_reg=0.1)
            conf_set_aps = cp.predict_raps(val_smx, adjusted_tau_aps, rng=True)

            # Gather set sizes by correctness for RAPS and APS
            correct_sizes_raps, incorrect_sizes_raps = gather_set_sizes_by_correctness(conf_set_raps, val_labels)
            correct_sizes_aps, incorrect_sizes_aps = gather_set_sizes_by_correctness(conf_set_aps, val_labels)

            # Convert set sizes to uncertainty levels
            uncertainty_data_raps_correct[model] = calculate_uncertainty(correct_sizes_raps)
            uncertainty_data_raps_incorrect[model] = calculate_uncertainty(incorrect_sizes_raps)
            uncertainty_data_aps_correct[model] = calculate_uncertainty(correct_sizes_aps)
            uncertainty_data_aps_incorrect[model] = calculate_uncertainty(incorrect_sizes_aps)

    # Plot trends for each combination
    plot_uncertainty_trends("(a) RAPS - Correct Predictions", uncertainty_data_raps_correct, models, color_map, "Uncertainty", "Frequency", "RAPS_correct_predictions.png")
    plot_uncertainty_trends("(b) RAPS - Incorrect Predictions", uncertainty_data_raps_incorrect, models, color_map, "Uncertainty", "Frequency", "RAPS_incorrect_predictions.png")
    plot_uncertainty_trends("(c) APS - Correct Predictions", uncertainty_data_aps_correct, models, color_map, "Uncertainty", "Frequency", "APS_correct_predictions.png")
    plot_uncertainty_trends("(d) APS - Incorrect Predictions", uncertainty_data_aps_incorrect, models, color_map, "Uncertainty", "Frequency", "APS_incorrect_predictions.png")

def plot_uncertainty_trends(title, uncertainty_data, models, color_map, xlabel, ylabel, filename):
    plt.figure(figsize=(12, 8))
    for model_idx, model in enumerate(models):
        color = color_map(model_idx)  # Get a unique color for each model
        uncertainties, counts = np.unique(uncertainty_data[model], return_counts=True)
        plt.plot(uncertainties, counts, marker='o', label=model, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Sample usage
if __name__ == '__main__':
    main(
        root_path='/Users/mehranzoravar/PycharmProjects/Music_Sarah/output',
        val_path='/Users/mehranzoravar/PycharmProjects/Music_Sarah/output',
        alpha=0.1,
        n_trials=10
    )