import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

models = {
    "DeiT": "./DeiT_output_test.npz",
    "Swin": "./Swin_output_test.npz",
    "ViT": "./ViT_output_test.npz",
    "Proposed Work": "./EACE_ViTs_output_test.npz"
}

results = []
for model_name, path in models.items():
    data = np.load(path)
    true_labels = data['labels']
    pred_labels = np.argmax(data['smx'], axis=1)

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_precision = precision_score(true_labels, pred_labels, average='macro')
    macro_recall = recall_score(true_labels, pred_labels, average='macro')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Macro Precision": round(macro_precision, 4),
        "Macro Recall": round(macro_recall, 4),
        "Macro F1": round(macro_f1, 4)
    })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))