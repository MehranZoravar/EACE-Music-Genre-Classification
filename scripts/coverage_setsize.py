from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import conformal_prediction as cp
import matplotlib
matplotlib.use('TkAgg')
# Function to compute entropy for each softmax output
def compute_entropy(softmax_output):
    """Compute the entropy for a given softmax output."""
    epsilon = 1e-12  # To avoid log(0)
    entropy = -np.sum(softmax_output * np.log(softmax_output + epsilon), axis=1)
    return entropy

# Function to adjust the threshold dynamically based on entropy
def adjust_threshold_based_on_entropy(base_tau, entropy, scaling_factor=0.2, num_classes=7):
    """Adjust the conformal threshold based on the normalized entropy of the softmax output."""
    max_entropy = np.log(num_classes)  # Maximum possible entropy for the given number of classes
    normalized_entropy = entropy / max_entropy  # Normalize entropy to [0, 1]
    adjusted_tau = base_tau + scaling_factor * normalized_entropy  # Adjust threshold
    return adjusted_tau

# Main function
def main(root_path: str, save_path: str, percent_cal: float, alpha: float, n_trials: int = 10):
    root_path = Path(root_path)
    save_path = Path(save_path)
    models = ['ViT', 'Swin', 'DeiT', 'EACE_ViTs']  # Customize your model names here
    color_map = plt.get_cmap("tab10")  # Use a color map for distinct colors

    # Dictionary to accumulate coverage results for each model and method
    coverage_results = defaultdict(lambda: defaultdict(dict))

    for trial in tqdm(range(n_trials), desc="Processing Trials"):
        print(f'Trial: {trial}')
        for model_idx, model in enumerate(models):
            npz_path = root_path / f'{model}_model' / f'{model}_output_test.npz'
            data = np.load(npz_path)
            smx = data['smx']
            labels = data['labels'].astype(int)

            # Split data into calibration and validation sets
            n = int(len(labels) * percent_cal)
            idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
            np.random.shuffle(idx)
            cal_smx, val_smx = smx[idx, :], smx[~idx, :]
            cal_labels, val_labels = labels[idx], labels[~idx]

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

            # Count set sizes and calculate coverage
            size_counts_raps = np.sum(conf_set_raps, axis=1)
            size_counts_aps = np.sum(conf_set_aps, axis=1)

            # Separate and calculate coverage for each unique set size
            coverage_by_size_raps = calculate_coverage_by_set_size(conf_set_raps, val_labels, size_counts_raps)
            coverage_by_size_aps = calculate_coverage_by_set_size(conf_set_aps, val_labels, size_counts_aps)

            # Store results
            coverage_results[model]['RAPS'] = coverage_by_size_raps
            coverage_results[model]['APS'] = coverage_by_size_aps

    # Plot RAPS results
    plt.figure(figsize=(10, 6))
    for model_idx, (model, methods) in enumerate(coverage_results.items()):
        if 'RAPS' in methods:
            coverage = methods['RAPS']
            color = color_map(model_idx)  # Get unique color for each model
            plot_coverage_vs_set_size(coverage, label=f"{model} - RAPS", color=color)

    plt.legend()
    plt.title("Coverage vs. Prediction Set Size for RAPS (All Models)")
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Coverage (%)")
    plt.grid(True)
    plt.show()

    # Plot APS results
    plt.figure(figsize=(10, 6))
    for model_idx, (model, methods) in enumerate(coverage_results.items()):
        if 'APS' in methods:
            coverage = methods['APS']
            color = color_map(model_idx)  # Use the same color for each model as in RAPS plot
            plot_coverage_vs_set_size(coverage, label=f"{model} - APS", color=color)

    plt.legend()
    plt.title("Coverage vs. Prediction Set Size for APS (All Models)")
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Coverage (%)")
    plt.grid(True)
    plt.show()

def calculate_coverage_by_set_size(conf_set, true_labels, set_sizes):
    """Calculate coverage for each unique set size."""
    unique_sizes = np.unique(set_sizes)
    coverage_by_size = {}
    for size in unique_sizes:
        indices = np.where(set_sizes == size)[0]
        correct_predictions = sum([1 if conf_set[i, true_labels[i]] == 1 else 0 for i in indices])
        coverage_by_size[size] = correct_predictions / len(indices) if len(indices) > 0 else 0
    return coverage_by_size

def plot_coverage_vs_set_size(coverage_dict, label, color):
    """Plot coverage vs. prediction set size."""
    sizes = sorted(coverage_dict.keys())
    coverages = [coverage_dict[size] for size in sizes]
    plt.plot(sizes, coverages, marker='o', label=label, color=color)

# Sample usage (adjust paths as necessary)
if __name__ == '__main__':
    main(
        root_path='../output',  # Update path to your softmax output files
        save_path='../Results',     # Update path where results will be saved
        percent_cal=0.5,                      # Proportion of data for calibration
        alpha=0.1,                             # Desired error level
        n_trials=5                          # Number of trials
    )
