import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Update rcParams for larger figures and fonts
plt.rcParams.update({
    'figure.figsize': (12, 10),  # Increase figure size
    'font.size': 18,           # Increase font size for all text
    'axes.titlesize': 16,      # Increase title font size
    'axes.labelsize': 16,      # Increase x and y label font size
    'xtick.labelsize': 16,     # Increase x-tick label font size
    'ytick.labelsize': 16,     # Increase y-tick label font size
    'legend.fontsize': 16      # Increase legend font size
})

# Load .npz file
data = np.load("./Swin_output_test.npz")  # Replace with the path to your .npz file
softmax_outputs = data['smx']
true_labels = data['labels']

# Get predicted labels by taking the argmax of the softmax outputs
pred_labels = np.argmax(softmax_outputs, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

# Define class names for music genres
class_names = ["blues", "classical", "country", "disco", "jazz", "metal", "pop", "reggae", "rock"]

# Count the number of occurrences for each class in true_labels
image_counts = [np.sum(true_labels == i) for i in range(len(class_names))]

# Plotting
plt.figure(figsize=(14, 12))  # Further increase figure size for better readability
sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap="Blues", cbar=True,
            xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})

# Add image counts and class names with colors to the y-axis labels
ax = plt.gca()
# Add image counts and class names to the y-axis labels
yticklabels = ax.get_yticklabels()
for i, label in enumerate(class_names):
    yticklabels[i].set_text(f"{label}:\n{image_counts[i]}")
ax.set_yticklabels(yticklabels, rotation=0, ha="right")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix with Class Information")

plt.tight_layout()  # Adjust layout to avoid clipping
plt.savefig('Swin_confusion_matrix.png', dpi=300)  # High-resolution save
plt.show()
