import argparse
import os
from prepare_pedestrians_yolo import prepare_cls_dataset
from convert_labelstudio_to_yolo import convert_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on the Pedestrians dataset (standing vs walking)."
    )
    parser.add_argument(
        "--mode",
        choices=["cls", "detect"],
        default="cls",
        help=(
            "'cls' trains YOLO-cls on Pedestrians/{standing,walking} crops (default). "
            "'detect' trains YOLO object detection using the Label Studio JSON + full-frame images."
        ),
    )
    parser.add_argument(
        "--pedestrians-dir",
        default="Pedestrians",
        help="Path to Pedestrians folder containing 'standing' and 'walking' subfolders",
    )
    parser.add_argument(
        "--cls-dataset-dir",
        default="yolo_cls_dataset",
        help="Output directory for the train/val split of Pedestrians crops",
    )
    parser.add_argument(
        "--json",
        default="project-7-at-2026-11-06-00-01-9e5699e9.json",
        help="Path to Label Studio JSON export (used when --mode detect)",
    )
    parser.add_argument(
        "--det-dataset-dir",
        default="yolo_dataset",
        help="Directory for converted YOLO detection dataset (used when --mode detect)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Pretrained YOLO weights (defaults to yolo11n-cls.pt for cls, yolo11n.pt for detect)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (224 for cls, 640 for detect)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.mode == "cls":
        train_dir = os.path.join(args.cls_dataset_dir, "train")
        if not os.path.isdir(train_dir):
            print(f"Preparing train/val split from '{args.pedestrians_dir}'...")
            prepare_cls_dataset(args.pedestrians_dir, args.cls_dataset_dir)

        model_name = args.model or "yolo11n-cls.pt"
        data_path = os.path.abspath(args.cls_dataset_dir)
        print(f"Training YOLO classification model ({model_name}) on {data_path}...")
        model = YOLO(model_name)
        model.train(
            data=data_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=os.path.abspath("runs/classify"),
            name="pedestrian_action",
            exist_ok=True,
        )
        print("Training complete! Best weights: runs/classify/pedestrian_action/weights/best.pt")

    else:
        if os.path.exists(args.json):
            convert_dataset(args.json, args.det_dataset_dir, args.pedestrians_dir)
        model_name = args.model or "yolo11n.pt"
        yaml_path = os.path.abspath(os.path.join(args.det_dataset_dir, "data.yaml"))
        imgsz = 640 if args.imgsz == 224 else args.imgsz
        print(f"Training YOLO detection model ({model_name}) on {yaml_path}...")
        model = YOLO(model_name)
        model.train(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            project=os.path.abspath("runs/detect"),
            name="pedestrian_action",
            exist_ok=True,
        )
        print("Training complete! Best weights: runs/detect/pedestrian_action/weights/best.pt")


if __name__ == "__main__":
    main()
