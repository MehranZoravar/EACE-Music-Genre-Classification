# 🎵 Entropy-Aware Conformal Ensemble of Vision Transformers for Music Genre Classification  

This repository contains the implementation of our framework **EACE-ViTs**, which applies **Vision Transformers (ViTs)** with **Conformal Prediction** to achieve robust and uncertainty-aware **music genre classification**.  
The framework leverages **spectrogram-based image representations** of audio, conformal calibration, and ensemble prediction to improve classification reliability.  

---

## 📌 Features  
- Convert raw audio to **Mel-spectrograms** (`mel_spectrogram.py`).  
- Create **custom dataset splits** for training, validation, calibration, and testing (`split_indices.py`).  
- Convert saved `.npy` files to `.npz` format (`npy_to_npz.py`).  
- Train **multiple ViT-based models** (ViT, DeiT, Swin) on spectrograms.  
- Apply **Conformal Prediction** to generate reliable prediction sets.  
- Evaluate performance with **accuracy, calibration, and coverage metrics**.  
- Easily extendable to other audio classification datasets.  

---

## 📂 Repository Structure  
```
EACE_ViTs_Music_Genres_Classification/
│── scripts/
│ ├── check_data_consistency.py # Validate dataset structure and files
│ ├── confusion_matrices.py # Generate confusion matrices
│ ├── coverage_setsize.py # Evaluate coverage vs. set size
│ ├── cp_experiments.py # Run conformal prediction experiments
│ ├── cp_extended_eval.py # Extended evaluation for CP
│ ├── eace.py # Entropy-Aware-Ensemble implementation
│ ├── mel_spectrogram.py # Convert audio to mel-spectrograms
│ ├── model_performance.py # Model performance evaluation
│ ├── save_npz.py # Save features/probabilities in .npz format
│ ├── save_splits.py # Create train/val/cal/test splits
│ ├── std_plot.py # Standard deviation plots
│ ├── test.py # Testing script
│ ├── train.py # Training script for Vision Transformers
│ └── uncertainty_plot.py # Uncertainty visualization
│
│── src/
│ ├── conformal_prediction.py # Core conformal prediction methods
│ └── evaluation.py # Evaluation utilities
│
│── Results/ # (Optional) Experimental results
│── requirements.txt # Dependencies (to be added)
```

---

## 🚀 Getting Started  

### 1. Clone the repository
```bash
git clone https://github.com/MehranZoravar/EACE_ViTs_Music_Genres_Classification.git
cd EACE_ViTs_Music_Genres_Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset  
- This project uses the **GTZAN Music Genre Dataset** (not included here due to size/license).  
- Download it manually and place it under:  
  ```
  ./Data/genres_original/
  ```
- Convert audio files into **Mel-spectrograms**:
  ```bash
  python mel_spectrogram.py
  ```

### 4. Generate dataset splits
```bash
python split_indices.py
```

### 5. Train models  
Example (training DeiT):
```bash
python train_vit.py --model deit_base --epochs 20 --batch_size 32
```

### 6. Run inference per model (save softmax + labels)
```bash
python test.py   # run once for test split → saves softmax_outputs_test.npy and labels_test.npy
python test.py   # run again for calibration split → saves softmax_outputs_cal.npy and labels_cal.npy
```
(Optional) Convert .npy files to .npz format:
```bash
python npy_npz.py
# Produces files like:
# ./output/DeiT_model/DeiT_output_test.npz
# ./output/DeiT_model/DeiT_output_cal.npz
# (repeat for ViT and Swin)
```

### 7. Build the EACE outputs
```bash
python eace.py
# Produces:
# ./output/EACE_ViTs_model/EACE_ViTs_output_test.npz
# ./output/EACE_ViTs_model/EACE_ViTs_output_cal.npz
```

### 8. Conformal Prediction experiments & CSVs
```bash
python cp_experiment.py      # Main CP experiments (averages + std over trials)
python cp_extened_eval.py    # Extended CP evaluation (correct vs. incorrect analysis)
# Results saved to ./Results/ (mean_results.csv, std_results.csv, etc.)
```

### 9. Model-level performance (accuracy / macro P / R / F1)
```bash
python model_performance.py
# Prints a table with Accuracy, Macro Precision, Recall, and F1
```

---

## 📊 Results  
Our experiments demonstrate that EACE-ViTs:  
- Improves **classification accuracy** over single models.  
- Provides **uncertainty-aware predictions** using Conformal Prediction.  
- Ensures high **coverage** while maintaining efficiency.

---

## ⚠️ Notes  
- **Datasets and saved models are excluded** from this repository.  
- Please download datasets separately and train models before reproducing results.  
- Pretrained model weights can be saved under `./saved_models/` (ignored via `.gitignore`).  

