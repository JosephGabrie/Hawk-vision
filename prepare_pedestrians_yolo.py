import argparse
import os
import random
import shutil

CLASSES = ["standing", "walking"]


def prepare_cls_dataset(
    source_dir: str = "Pedestrians",
    output_dir: str = "yolo_cls_dataset",
    val_split: float = 0.2,
    balance: bool = True,
    seed: int = 42,
):
    """
    Organizes Pedestrians/{standing,walking} into Ultralytics YOLO classification format.
    When balance=True, equalizes the class counts between 'standing' and 'walking' so the
    model does not over-predict 'walking' due to the 5:1 class imbalance in Pedestrians/.
    """
    random.seed(seed)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    class_files = {}
    for cls_name in CLASSES:
        cls_src = os.path.join(source_dir, cls_name)
        if not os.path.isdir(cls_src):
            raise FileNotFoundError(f"Expected class folder not found: {cls_src}")

        files = [
            f for f in os.listdir(cls_src)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        files.sort()
        random.shuffle(files)
        class_files[cls_name] = files

    if balance:
        min_count = min(len(v) for v in class_files.values())
        print(f"Balancing classes to {min_count} images per class (1:1 standing vs walking)...")
        for cls_name in CLASSES:
            class_files[cls_name] = class_files[cls_name][:min_count]

    total_train = 0
    total_val = 0

    for cls_name in CLASSES:
        files = class_files[cls_name]
        val_count = int(len(files) * val_split)
        val_files = files[:val_count]
        train_files = files[val_count:]

        cls_src = os.path.join(source_dir, cls_name)
        for split_name, split_files in [("train", train_files), ("val", val_files)]:
            dst_dir = os.path.join(output_dir, split_name, cls_name)
            os.makedirs(dst_dir, exist_ok=True)
            for fn in split_files:
                shutil.copy2(os.path.join(cls_src, fn), os.path.join(dst_dir, fn))

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"Class '{cls_name}': {len(train_files)} train, {len(val_files)} val images")

    print(f"\nDataset ready at '{os.path.abspath(output_dir)}' ({total_train} train, {total_val} val).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Pedestrians folder for YOLO training.")
    parser.add_argument("--source", default="Pedestrians", help="Path to Pedestrians folder")
    parser.add_argument("--output", default="yolo_cls_dataset", help="Output dataset directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable 1:1 class balancing between standing and walking",
    )
    args = parser.parse_args()
    prepare_cls_dataset(args.source, args.output, args.val_split, balance=not args.no_balance)
