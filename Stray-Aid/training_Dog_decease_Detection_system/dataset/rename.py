"""
rename_dataset.py
-----------------
Renames all images in the dataset (train / valid / test) to a clean,
consistent format:
    <split>_<class>_<zero-padded-index>.<ext>

Example:
    train/demodicosis/train_demodicosis_0001.jpg
    valid/Healthy/valid_Healthy_0003.png

Usage:
    python rename_dataset.py
    python rename_dataset.py --base_dir "C:/Users/Shailendra/Downloads/Final_Proj/Stray-Aid/training_Dog_decease_Detection_system/dataset"
    python rename_dataset.py --dry_run          # preview without renaming
"""

import os
import argparse
from pathlib import Path

# ── Supported image extensions ──────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# ── Default base directory ───────────────────────────────────────────────────
DEFAULT_BASE = (
    r"C:\Users\Shailendra\Downloads\Final_Proj\Stray-Aid"
    r"\training_Dog_decease_Detection_system\dataset"
)

# ── Classes (must match folder names exactly) ────────────────────────────────
CLASSES = [
    "demodicosis",
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "ringworm",
]

SPLITS = ["train", "valid", "test"]


def rename_split(base: Path, split: str, dry_run: bool) -> int:
    """Rename all images inside base/<split>/<class>/ folders."""
    total = 0
    split_dir = base / split

    if not split_dir.exists():
        print(f"  [SKIP] {split_dir} does not exist.")
        return 0

    for cls in CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            print(f"  [SKIP] {cls_dir} does not exist.")
            continue

        # Collect only image files, sorted for reproducibility
        images = sorted(
            [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        )

        print(f"\n  [{split}/{cls}]  {len(images)} images found")

        for idx, img_path in enumerate(images, start=1):
            ext = img_path.suffix.lower()
            new_name = f"{split}_{cls}_{idx:04d}{ext}"
            new_path = cls_dir / new_name

            if img_path.name == new_name:
                print(f"    (already named) {img_path.name}")
                continue

            # Avoid collision: if target exists, use a temp name first
            if dry_run:
                print(f"    {img_path.name}  →  {new_name}")
            else:
                if new_path.exists() and new_path != img_path:
                    tmp = cls_dir / f"__tmp_{idx:04d}{ext}"
                    img_path.rename(tmp)
                    tmp.rename(new_path)
                else:
                    img_path.rename(new_path)
                print(f"    ✓  {img_path.name}  →  {new_name}")

            total += 1

    return total


def main():
    parser = argparse.ArgumentParser(description="Rename dog-disease dataset images.")
    parser.add_argument(
        "--base_dir",
        default=DEFAULT_BASE,
        help="Path to the 'dataset' folder (contains train/valid/test).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview renames without actually changing files.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f"[ERROR] Base directory not found:\n  {base}")
        return

    print("=" * 60)
    print("  Dog Disease Dataset — Image Renaming Script")
    print("=" * 60)
    print(f"  Base : {base}")
    print(f"  Mode : {'DRY RUN (no changes)' if args.dry_run else 'LIVE'}")
    print("=" * 60)

    grand_total = 0
    for split in SPLITS:
        print(f"\n[SPLIT: {split.upper()}]")
        grand_total += rename_split(base, split, args.dry_run)

    print("\n" + "=" * 60)
    action = "Would rename" if args.dry_run else "Renamed"
    print(f"  {action} {grand_total} file(s) in total.")
    print("=" * 60)


if __name__ == "__main__":
    main()