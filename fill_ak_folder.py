"""
fill_ak_folder.py

Your data\\test\\AK folder came up empty. This script fills ONLY that one
folder, using the ISIC_2019_Training_GroundTruth.csv you already have in
this project folder, plus your raw ISIC image folder.

EDIT THE ONE PATH BELOW (where your raw, unsorted ISIC images live -- the
big folder of all .jpg files, not the data\test folders), then run:

    python fill_ak_folder.py
"""

import os
import csv
import random
import shutil

# --- EDIT THIS ONE PATH ---
RAW_IMAGE_DIR = r"C:\Users\tulik\Downloads\archive.zip\Skin cancer ISIC The International Skin Imaging Collaboration"   # folder of ALL raw .jpg images
# ---------------------------

LABELS_CSV = "ISIC_2019_Training_GroundTruth.csv"   # already in this folder
OUTPUT_DIR = os.path.join("data", "test", "AK")
TEST_FRACTION = 0.15
RANDOM_SEED = 42

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def find_image_path(image_id):
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(RAW_IMAGE_DIR, image_id + ext)
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(RAW_IMAGE_DIR, image_id)
    if os.path.exists(candidate):
        return candidate
    return None


def main():
    random.seed(RANDOM_SEED)

    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"ERROR: RAW_IMAGE_DIR does not exist: {RAW_IMAGE_DIR}")
        print("Edit the path at the top of this script and try again.")
        return
    if not os.path.exists(LABELS_CSV):
        print(f"ERROR: Can't find {LABELS_CSV} in this folder.")
        return

    with open(LABELS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    id_col = None
    for candidate in ["image", "image_id", "ISIC_ID", "isic_id"]:
        if candidate in fieldnames:
            id_col = candidate
            break
    if id_col is None:
        id_col = fieldnames[0]

    ak_col = None
    for candidate in ["AK", "AKIEC", "ak", "akiec"]:
        if candidate in fieldnames:
            ak_col = candidate
            break
    if ak_col is None:
        print(f"ERROR: Couldn't find an AK/AKIEC column. Detected columns "
              f"in your CSV: {fieldnames}")
        print("Tell me the exact column name for Actinic Keratosis and "
              "I'll adjust this script.")
        return

    print(f"Using image-id column: '{id_col}', AK column: '{ak_col}'")

    ak_ids = []
    for row in rows:
        try:
            val = float(row[ak_col])
        except (ValueError, KeyError):
            continue
        if val >= 0.5:  # one-hot, so this means "this row IS Actinic Keratosis"
            ak_ids.append(row[id_col].strip())

    print(f"Found {len(ak_ids)} AK-labeled rows in the CSV.")
    if not ak_ids:
        print("No AK rows found at all -- something's off with the column "
              "name or the CSV itself. Nothing to copy.")
        return

    random.shuffle(ak_ids)
    n_test = max(1, int(len(ak_ids) * TEST_FRACTION))
    test_ids = ak_ids[:n_test]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    copied, missing = 0, 0
    for image_id in test_ids:
        src = find_image_path(image_id)
        if src is None:
            missing += 1
            continue
        dst = os.path.join(OUTPUT_DIR, os.path.basename(src))
        shutil.copyfile(src, dst)
        copied += 1

    print(f"\nCopied {copied} AK images into {OUTPUT_DIR}")
    if missing:
        print(f"WARNING: {missing} image ids from the CSV weren't found in "
              f"RAW_IMAGE_DIR -- double check that path is correct.")
    print("\nNow run: python evaluate.py")


if __name__ == "__main__":
    main()
