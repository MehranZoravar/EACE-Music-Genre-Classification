import os

# Define dataset paths
IMAGE_PATH = "./Data/images_original"
SPEC_PATH = "./mel_spectrogram"

# Get filenames in both directories
img_filenames = set()
spec_filenames = set()

# Collect all image filenames (without extensions)
for genre in os.listdir(IMAGE_PATH):
    genre_path = os.path.join(IMAGE_PATH, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            if file.endswith((".jpg", ".png")):  # Adjust if needed
                img_filenames.add(file.split('.')[0])  # Store only filename without extension

# Collect all spectrogram filenames (without extensions)
for genre in os.listdir(SPEC_PATH):
    genre_path = os.path.join(SPEC_PATH, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            if file.endswith(".png"):  # Spectrograms are usually PNG
                spec_filenames.add(file.split('.')[0])

# Find missing spectrograms and images
missing_specs = img_filenames - spec_filenames
missing_imgs = spec_filenames - img_filenames

# Report missing files
print(f"✅ Total images: {len(img_filenames)}")
print(f"✅ Total spectrograms: {len(spec_filenames)}")
print(f"⚠️ Missing spectrograms: {len(missing_specs)}")
print(f"⚠️ Missing images: {len(missing_imgs)}")

# Log missing files for reference
if missing_specs:
    print("❌ The following spectrograms are missing:")
    print(missing_specs)

if missing_imgs:
    print("❌ The following images are missing:")
    print(missing_imgs)

def rename_files_to_match(image_path, spec_path):
    """ Rename files in the spectrogram folder to match images if needed. """
    for genre in os.listdir(image_path):
        img_genre_path = os.path.join(image_path, genre)
        spec_genre_path = os.path.join(spec_path, genre)

        if not os.path.isdir(img_genre_path) or not os.path.isdir(spec_genre_path):
            continue

        # Get all image files
        img_files = [f for f in os.listdir(img_genre_path) if f.endswith((".jpg", ".png"))]

        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            expected_spec_file = f"{base_name}.png"

            spec_full_path = os.path.join(spec_genre_path, expected_spec_file)
            if not os.path.exists(spec_full_path):
                print(f"⚠️ Renaming missing spectrogram: {expected_spec_file}")
                # Rename spectrogram file manually if required (Modify this logic based on your mismatch)
                # os.rename(old_spec_path, spec_full_path)


# Run the renaming function
rename_files_to_match(IMAGE_PATH, SPEC_PATH)
