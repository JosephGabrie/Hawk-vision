import argparse
import json
import os
import random
import shutil
from collections import Counter, deque

# Label mapping: merge all walking directions into 'walking'
CLASS_NAMES = ["standing", "walking"]
LABEL_TO_CLASS_ID = {
    "Standing": 0,
    "Walking perpendicular left": 1,
    "Walking perpendicular right": 1,
    "Walking towards": 1,
    "Walking away": 1,
}


class InputStream:
    def __init__(self, data: str):
        self.data = data
        self.i = 0

    def read(self, size: int) -> int:
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2) if out else 0


def bytes2bit(data: list[int]) -> str:
    # Fast byte-to-bitstring conversion using lookup table
    return "".join(f"{b:08b}" for b in data)


def decode_rle(rle: list[int]) -> list[int]:
    """Decode Label Studio brush RLE array into flat RGBA array (length = w * h * 4)."""
    input_stream = InputStream(bytes2bit(rle))
    num = input_stream.read(32)
    word_size = input_stream.read(5) + 1
    rle_sizes = [input_stream.read(4) + 1 for _ in range(4)]
    out = [0] * num
    i = 0
    while i < num:
        x = input_stream.read(1)
        j = i + 1 + input_stream.read(rle_sizes[input_stream.read(2)])
        if x:
            val = input_stream.read(word_size)
            limit = min(j, num)
            out[i:limit] = [val] * (limit - i)
            i = j
        else:
            limit = min(j, num)
            while i < limit:
                out[i] = input_stream.read(word_size)
                i += 1
    return out


def extract_bboxes_from_mask(rle: list[int], width: int, height: int) -> list[tuple[int, int, int, int]]:
    """
    Extract individual pedestrian bounding boxes (xmin, ymin, xmax, ymax) from a Label Studio RLE mask.
    In Label Studio brush annotations, multiple pedestrians of the same class in one image are sometimes
    painted in a single brush layer. We use connected-component clustering on a coarse grid (cell=4px)
    with a generous vertical bridge and modest horizontal bridge so a single person's brush strokes stay
    together while distinct pedestrians are separated.
    """
    out = decode_rle(rle)
    cell = 4
    gw = (width + cell - 1) // cell
    gh = (height + cell - 1) // cell

    # Track exact pixel bounds per coarse cell
    cell_bounds = {}  # (gy, gx) -> [xmin, ymin, xmax, ymax, count]
    alpha_slice = out[3::4]
    if not any(alpha_slice):
        alpha_slice = out[0::4]

    for pix_idx, val in enumerate(alpha_slice):
        if val > 0:
            y = pix_idx // width
            x = pix_idx % width
            key = (y // cell, x // cell)
            if key not in cell_bounds:
                cell_bounds[key] = [x, y, x, y, 1]
            else:
                b = cell_bounds[key]
                if x < b[0]:
                    b[0] = x
                if y < b[1]:
                    b[1] = y
                if x > b[2]:
                    b[2] = x
                if y > b[3]:
                    b[3] = y
                b[4] += 1

    if not cell_bounds:
        return []

    visited = set()
    raw_comps = []

    # Bridge gaps up to ~12px horizontally and ~24px vertically (in 4px grid units: dx<=3, dy<=6)
    for start_key in cell_bounds:
        if start_key in visited:
            continue
        q = deque([start_key])
        visited.add(start_key)
        xmin, ymin, xmax, ymax, total_px = cell_bounds[start_key]
        while q:
            cy, cx = q.popleft()
            for dy in range(-6, 7):
                for dx in range(-3, 4):
                    nk = (cy + dy, cx + dx)
                    if nk in cell_bounds and nk not in visited:
                        visited.add(nk)
                        q.append(nk)
                        b = cell_bounds[nk]
                        if b[0] < xmin:
                            xmin = b[0]
                        if b[1] < ymin:
                            ymin = b[1]
                        if b[2] > xmax:
                            xmax = b[2]
                        if b[3] > ymax:
                            ymax = b[3]
                        total_px += b[4]

        # Filter tiny stray brush dots (< 50 pixels)
        if total_px >= 50 and (xmax - xmin) >= 10 and (ymax - ymin) >= 20:
            raw_comps.append((xmin, ymin, xmax, ymax, total_px))

    if not raw_comps:
        # Fallback to overall bounding box if all components were small
        xmin = min(b[0] for b in cell_bounds.values())
        ymin = min(b[1] for b in cell_bounds.values())
        xmax = max(b[2] for b in cell_bounds.values())
        ymax = max(b[3] for b in cell_bounds.values())
        return [(xmin, ymin, xmax, ymax)]

    return [(c[0], c[1], c[2], c[3]) for c in raw_comps]


def convert_dataset(
    json_path: str,
    output_dir: str,
    images_source_dir: str | None = None,
    val_split: float = 0.2,
    seed: int = 42,
):
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Index available source images if provided
    image_lookup = {}
    if images_source_dir and os.path.isdir(images_source_dir):
        for root, _, files in os.walk(images_source_dir):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    image_lookup[fn] = os.path.join(root, fn)
                    # Also index without Label Studio 8-char prefix if present
                    if "-" in fn and len(fn.split("-", 1)[0]) == 8:
                        image_lookup[fn.split("-", 1)[1]] = os.path.join(root, fn)

    records = []
    class_counts = Counter()
    multi_comp_results = 0

    for task in tasks:
        file_upload = task.get("file_upload", "")
        orig_filename = file_upload.split("-", 1)[1] if "-" in file_upload else file_upload
        stem = os.path.splitext(orig_filename)[0]

        yolo_lines = []
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            for res in ann.get("result", []):
                if res.get("type") != "brushlabels":
                    continue
                val = res.get("value", {})
                labels = val.get("brushlabels", [])
                rle = val.get("rle", [])
                w = res.get("original_width")
                h = res.get("original_height")
                if not labels or not rle or not w or not h:
                    continue

                raw_label = labels[0]
                if raw_label not in LABEL_TO_CLASS_ID:
                    print(f"Warning: unknown label '{raw_label}', skipping")
                    continue
                class_id = LABEL_TO_CLASS_ID[raw_label]

                bboxes = extract_bboxes_from_mask(rle, w, h)
                if len(bboxes) > 1:
                    multi_comp_results += 1

                for xmin, ymin, xmax, ymax in bboxes:
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    xc = (xmin + xmax) / (2.0 * w)
                    yc = (ymin + ymax) / (2.0 * h)
                    # Clamp to [0, 1]
                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    class_counts[CLASS_NAMES[class_id]] += 1

        records.append({
            "file_upload": file_upload,
            "orig_filename": orig_filename,
            "stem": stem,
            "yolo_lines": yolo_lines,
        })

    random.seed(seed)
    shuffled = list(records)
    random.shuffle(shuffled)
    val_count = int(len(shuffled) * val_split)
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]

    for split_name, split_recs in [("train", train_records), ("val", val_records)]:
        lbl_dir = os.path.join(output_dir, "labels", split_name)
        img_dir = os.path.join(output_dir, "images", split_name)
        os.makedirs(lbl_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)

        copied_images = 0
        for rec in split_recs:
            lbl_path = os.path.join(lbl_dir, f"{rec['stem']}.txt")
            with open(lbl_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(rec["yolo_lines"]) + ("\n" if rec["yolo_lines"] else ""))

            # Copy image if found in images_source_dir
            src_img = image_lookup.get(rec["orig_filename"]) or image_lookup.get(rec["file_upload"])
            if src_img and os.path.exists(src_img):
                ext = os.path.splitext(src_img)[1]
                dst_img = os.path.join(img_dir, f"{rec['stem']}{ext}")
                shutil.copy2(src_img, dst_img)
                copied_images += 1

        print(f"Split '{split_name}': {len(split_recs)} label files written, {copied_images} images copied.")

    # Write Ultralytics data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_output_dir = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = (
        f"path: {abs_output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: standing\n"
        f"  1: walking\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"\nConversion complete!")
    print(f"Total tasks processed: {len(records)}")
    print(f"Multi-instance brush layers separated: {multi_comp_results}")
    print(f"Bounding boxes per class: {dict(class_counts)}")
    print(f"YOLO dataset config saved to: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Label Studio RLE JSON to YOLO detection dataset.")
    parser.add_argument(
        "--json",
        default="project-7-at-2026-11-06-00-01-9e5699e9.json",
        help="Path to Label Studio JSON export",
    )
    parser.add_argument(
        "--output",
        default="yolo_dataset",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory containing FudanPed*.png images to copy into yolo_dataset/images/{train,val}",
    )
    args = parser.parse_args()
    convert_dataset(args.json, args.output, args.images_dir)
