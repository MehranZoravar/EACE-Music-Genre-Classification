import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Audio processing parameters
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512

# Define dataset paths
DATASET_PATH = "./Data/genres_original"
OUTPUT_PATH = "./mel_spectrogram"

# Ensure the output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# List of genres in the dataset
genres = [g for g in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, g))]

def audio_to_mel_spectrogram(audio_path, save_path):
    """Convert an audio file to an RGB Mel-Spectrogram image."""

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=30)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot and save as RGB image
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)  # Adjust size to Swin Transformer input
    ax.axis('off')
    librosa.display.specshow(mel_spec_db, sr=sr, cmap="viridis")

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Process all audio files
for genre in tqdm(genres, desc="Processing genres"):
    genre_path = os.path.join(DATASET_PATH, genre)
    save_genre_path = os.path.join(OUTPUT_PATH, genre)
    os.makedirs(save_genre_path, exist_ok=True)

    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            file_path = os.path.join(genre_path, file)

            # Fix filename formatting: Remove the extra dot
            base_name = file.replace(".wav", "")  # Remove extension
            base_name = base_name.replace(".", "")  # Remove extra dots

            save_file_path = os.path.join(save_genre_path, base_name + ".png")

            audio_to_mel_spectrogram(file_path, save_file_path)

print("✅ Mel-spectrograms saved at:", OUTPUT_PATH)
