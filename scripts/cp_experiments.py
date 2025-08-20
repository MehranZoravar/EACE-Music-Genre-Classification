from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from src import conformal_prediction as cp, evaluation

# Function to compute entropy for each softmax output
def compute_entropy(softmax_output):
    """Compute the entropy for a given softmax output."""
    epsilon = 1e-12  # To avoid log(0)
    entropy = -np.sum(softmax_output * np.log(softmax_output + epsilon), axis=1)
    return entropy

# Function to adjust the threshold dynamically based on entropy
def adjust_threshold_based_on_entropy(base_tau, entropy, scaling_factor=0.2, num_classes=10):
    """Adjust the conformal threshold based on the normalized entropy of the softmax output."""
    max_entropy = np.log(num_classes)  # Maximum possible entropy for the given number of classes
    normalized_entropy = entropy / max_entropy  # Normalize entropy to [0, 1]
    adjusted_tau = base_tau + scaling_factor * normalized_entropy  # Adjust threshold
    return adjusted_tau

def main(root_path: str, save_path: str, percent_cal: float, alpha: float, n_trials: int = 10):
    """
    Performs conformal prediction and evaluation on softmax outputs with entropy-based threshold adjustment.
    Metrics are averaged across n_trials random splits, and the resulting mean & std are saved to .csv files.
    :param root_path: Root path where softmax scores are stored
    :param save_path: Where to save results
    :param percent_cal: Percent of the data to use for calibration
    :param alpha: Desired error level
    :param n_trials: Number of trials to run
    """
    root_path = Path(root_path)
    save_path = Path(save_path)

    results_list = []  # List to keep track of results across multiple trials
    models = ['ViT', 'Swin', 'DeiT', 'EACE_ViTs']  # Customize your model names here

    for trial in tqdm(range(n_trials)):  # run multiple trials
        print(f'Trial: {trial}')

        for model in models:
            results = defaultdict(list)

            # Load the npz file for the current model
            npz_path = root_path / f'{model}_model' / f'{model}_output_cal.npz'
            data = np.load(npz_path)
            smx = data['smx']  # Softmax scores
            labels = data['labels'].astype(int)

            # Split the softmax scores into calibration and validation sets
            n = int(len(labels) * percent_cal)
            idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
            np.random.shuffle(idx)
            cal_smx, val_smx = smx[idx, :], smx[~idx, :]
            cal_labels, val_labels = labels[idx], labels[~idx]

            # Evaluate accuracy
            acc = evaluation.compute_accuracy(val_smx, val_labels)

            # Compute entropy for calibration and validation sets
            cal_entropy = compute_entropy(cal_smx)
            val_entropy = compute_entropy(val_smx)

            # Get regularization penalties for RAPS (use values specific to your model or keep it generic)
            k_reg = 2
            lambda_reg = 0.1

            # Calibrate on calibration set using standard conformal prediction
            tau_thr = cp.calibrate_threshold(cal_smx, cal_labels, alpha)
            tau_raps = cp.calibrate_raps_adjusted(cal_smx, cal_labels, alpha=alpha, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
            tau_aps = cp.calibrate_raps_adjusted(cal_smx, cal_labels, alpha=alpha, rng=True, k_reg=None, lambda_reg=None)

            # Now, adjust tau_thr based on entropy (this is the new part!)
            adjusted_tau_thr = adjust_threshold_based_on_entropy(tau_thr, val_entropy, scaling_factor=0.1)
            adjusted_tau_raps = adjust_threshold_based_on_entropy(tau_raps, val_entropy, scaling_factor=0.1)
            adjusted_tau_aps = adjust_threshold_based_on_entropy(tau_aps, val_entropy, scaling_factor=0.1)

            # Get confidence sets using entropy-adjusted threshold for validation set
            conf_set_thr = cp.predict_threshold(val_smx, adjusted_tau_thr)
            conf_set_raps = cp.predict_raps(val_smx, adjusted_tau_raps, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
            conf_set_aps = cp.predict_raps(val_smx, adjusted_tau_aps, rng=True)

            # Evaluate coverage
            cov_thr = float(evaluation.compute_coverage(conf_set_thr, val_labels))
            cov_raps = float(evaluation.compute_coverage(conf_set_raps, val_labels))
            cov_aps = float(evaluation.compute_coverage(conf_set_aps, val_labels))

            # Evaluate set size
            size_thr, _ = evaluation.compute_size(conf_set_thr)
            size_raps, _ = evaluation.compute_size(conf_set_raps)
            size_aps, _ = evaluation.compute_size(conf_set_aps)

            # Save results for this trial and model
            results['model'].append(model)
            results['acc'].append(acc)
            results['cov_thr'].append(cov_thr)
            results['cov_raps'].append(cov_raps)
            results['cov_aps'].append(cov_aps)
            results['size_thr'].append(float(size_thr))
            results['size_raps'].append(float(size_raps))
            results['size_aps'].append(float(size_aps))

            # Append trial results to the corresponding list
            results_list.append(pd.DataFrame.from_dict(results))

    # Save results to CSV files
    concatenated = pd.concat(results_list)  # concatenate all trial results
    avg = concatenated.groupby('model').mean()
    std = concatenated.groupby('model').std()

    # Save results
    avg_file = save_path / 'mean_results.csv'
    avg_file.parent.mkdir(parents=True, exist_ok=True)
    avg.to_csv(avg_file)

    std_file = save_path / 'std_results.csv'
    std_file.parent.mkdir(parents=True, exist_ok=True)
    std.to_csv(std_file)

if __name__ == '__main__':
    main(
        root_path='../output',  # Directory with softmax outputs
        save_path='../Results',  # Directory to save results
        percent_cal=0.5,  # 50% calibration
        alpha=0.1,  # Error level
        n_trials=10  # Number of trials
    )
