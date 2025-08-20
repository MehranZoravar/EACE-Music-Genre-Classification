import numpy as np

# Load the saved softmax outputs for all three models
vit_data = np.load('./vit_output_test.npz')
deit_data = np.load('./deit_output_test.npz')
swin_data = np.load('./swin_output_test.npz')

# Extract the softmax outputs from the 'smx' key and the labels (if needed)
softmax_outputs_vit = vit_data['smx']
softmax_outputs_deit = deit_data['smx']
softmax_outputs_swin = swin_data['smx']
labels = deit_data['labels']  # Assuming labels are the same across files

# Define function to calculate entropy
def calculate_entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probabilities * np.log(probabilities))

# Prepare for weighted ensemble
high_weight = 0.8
low_weight = 0.1
all_weighted_ensemble_probs = []

# Iterate over each sample in the softmax outputs
for i in range(len(softmax_outputs_vit)):
    # Get the softmax outputs for the current sample from each model
    probs_vit = softmax_outputs_vit[i]
    probs_deit = softmax_outputs_deit[i]
    probs_swin = softmax_outputs_swin[i]

    # Calculate entropy for the current sample
    entropy_vit = calculate_entropy(probs_vit)
    entropy_deit = calculate_entropy(probs_deit)
    entropy_swin = calculate_entropy(probs_swin)

    # Stack entropies and select the lowest entropy for weighting
    entropies = np.array([entropy_vit, entropy_deit, entropy_swin])
    min_entropy_index = np.argmin(entropies)

    # Apply weights based on entropy
    if min_entropy_index == 0:
        weighted_probs = (probs_vit * high_weight +
                          probs_deit * low_weight +
                          probs_swin * low_weight)
    elif min_entropy_index == 1:
        weighted_probs = (probs_vit * low_weight +
                          probs_deit * high_weight +
                          probs_swin * low_weight)
    else:
        weighted_probs = (probs_vit * low_weight +
                          probs_deit * low_weight +
                          probs_swin * high_weight)

    # Save the weighted probabilities for this sample
    all_weighted_ensemble_probs.append(weighted_probs)

# Optionally save the results to a new .npz file
np.savez('./EACE_ViTs_output_cal.npz', labels=labels, smx=all_weighted_ensemble_probs)
print("Weighted ensemble probabilities saved to 'EACE-ViTs_softmax_outputs.npz'")