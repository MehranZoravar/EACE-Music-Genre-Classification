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
            npz_path = root_path / f'{model}_model' / f'{model}_output_test.npz'
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

            # Compute predicted labels and separate val_smx into correct and incorrect predictions
            predicted_labels = np.argmax(val_smx, axis=1)
            correct_mask = predicted_labels == val_labels
            incorrect_mask = ~correct_mask
            correct_val_smx, incorrect_val_smx = val_smx[correct_mask], val_smx[incorrect_mask]
            correct_val_labels, incorrect_val_labels = val_labels[correct_mask], val_labels[incorrect_mask]

            # Compute entropy for calibration and validation sets
            cal_entropy = compute_entropy(cal_smx)
            correct_val_entropy = compute_entropy(correct_val_smx)
            incorrect_val_entropy = compute_entropy(incorrect_val_smx)

            # Get regularization penalties for RAPS
            k_reg = 2
            lambda_reg = 0.1

            # Calibrate on calibration set using standard conformal prediction
            tau_thr = cp.calibrate_threshold(cal_smx, cal_labels, alpha)
            tau_raps = cp.calibrate_raps_adjusted(cal_smx, cal_labels, alpha=alpha, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
            tau_aps = cp.calibrate_raps_adjusted(cal_smx, cal_labels, alpha=alpha, rng=True, k_reg=None, lambda_reg=None)

            # Adjust tau_thr based on entropy for both correct and incorrect predictions
            adjusted_tau_thr_correct = adjust_threshold_based_on_entropy(tau_thr, correct_val_entropy, scaling_factor=0.1)
            adjusted_tau_thr_incorrect = adjust_threshold_based_on_entropy(tau_thr, incorrect_val_entropy, scaling_factor=0.1)
            adjusted_tau_raps_correct = adjust_threshold_based_on_entropy(tau_raps, correct_val_entropy, scaling_factor=0.1)
            adjusted_tau_raps_incorrect = adjust_threshold_based_on_entropy(tau_raps, incorrect_val_entropy, scaling_factor=0.1)
            adjusted_tau_aps_correct = adjust_threshold_based_on_entropy(tau_aps, correct_val_entropy, scaling_factor=0.1)
            adjusted_tau_aps_incorrect = adjust_threshold_based_on_entropy(tau_aps, incorrect_val_entropy, scaling_factor=0.1)

            # Get confidence sets using entropy-adjusted threshold for correct and incorrect predictions
            conf_set_thr_correct = cp.predict_threshold(correct_val_smx, adjusted_tau_thr_correct)
            conf_set_thr_incorrect = cp.predict_threshold(incorrect_val_smx, adjusted_tau_thr_incorrect)
            conf_set_raps_correct = cp.predict_raps(correct_val_smx, adjusted_tau_raps_correct, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
            conf_set_raps_incorrect = cp.predict_raps(incorrect_val_smx, adjusted_tau_raps_incorrect, rng=True, k_reg=k_reg, lambda_reg=lambda_reg)
            conf_set_aps_correct = cp.predict_raps(correct_val_smx, adjusted_tau_aps_correct, rng=True)
            conf_set_aps_incorrect = cp.predict_raps(incorrect_val_smx, adjusted_tau_aps_incorrect, rng=True)

            # Evaluate coverage and size for correct and incorrect predictions
            cov_thr_correct = float(evaluation.compute_coverage(conf_set_thr_correct, correct_val_labels))
            cov_thr_incorrect = float(evaluation.compute_coverage(conf_set_thr_incorrect, incorrect_val_labels))
            cov_raps_correct = float(evaluation.compute_coverage(conf_set_raps_correct, correct_val_labels))
            cov_raps_incorrect = float(evaluation.compute_coverage(conf_set_raps_incorrect, incorrect_val_labels))
            cov_aps_correct = float(evaluation.compute_coverage(conf_set_aps_correct, correct_val_labels))
            cov_aps_incorrect = float(evaluation.compute_coverage(conf_set_aps_incorrect, incorrect_val_labels))

            size_thr_correct, _ = evaluation.compute_size(conf_set_thr_correct)
            size_thr_incorrect, _ = evaluation.compute_size(conf_set_thr_incorrect)
            size_raps_correct, _ = evaluation.compute_size(conf_set_raps_correct)
            size_raps_incorrect, _ = evaluation.compute_size(conf_set_raps_incorrect)
            size_aps_correct, _ = evaluation.compute_size(conf_set_aps_correct)
            size_aps_incorrect, _ = evaluation.compute_size(conf_set_aps_incorrect)

            # Save results for this trial and model
            results['model'].append(model)
            results['acc'].append(acc)
            #results['cov_thr_correct'].append(cov_thr_correct)
            #results['cov_thr_incorrect'].append(cov_thr_incorrect)
            results['cov_raps_correct'].append(cov_raps_correct)
            results['cov_raps_incorrect'].append(cov_raps_incorrect)
            results['cov_aps_correct'].append(cov_aps_correct)
            results['cov_aps_incorrect'].append(cov_aps_incorrect)
            #results['size_thr_correct'].append(float(size_thr_correct))
            #results['size_thr_incorrect'].append(float(size_thr_incorrect))
            results['size_raps_correct'].append(float(size_raps_correct))
            results['size_raps_incorrect'].append(float(size_raps_incorrect))
            results['size_aps_correct'].append(float(size_aps_correct))
            results['size_aps_incorrect'].append(float(size_aps_incorrect))

            # Append trial results to the corresponding list
            results_list.append(pd.DataFrame.from_dict(results))

    # Save results to CSV files
    concatenated = pd.concat(results_list)  # concatenate all trial results
    avg = concatenated.groupby('model').mean()
    std = concatenated.groupby('model').std()

    # Save results
    avg_file = save_path / 'mean_results_correct_incorrect.csv'
    avg_file.parent.mkdir(parents=True, exist_ok=True)
    avg.to_csv(avg_file)

    std_file = save_path / 'std_results_correct_incorrect.csv'
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
