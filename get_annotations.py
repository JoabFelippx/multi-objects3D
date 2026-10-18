import json
import os
from pathlib import Path

import numpy as np
import cv2

from ultralytics import YOLO

from utils import CALIB_PATH_RAW, IMAGES_PATH_RAW, ANNOTATIONS_PATH, PROJECT_ROOT


def load_cameras_params(cameras):
    calib_params_by_cam = {}

    for cam in cameras:
        path = CALIB_PATH_RAW / f"calib_rt{cam}.npz"

        data = np.load(path)
        calib_params_by_cam[cam] = data

    return calib_params_by_cam


def load_images(cameras):
    images_by_cam = {}

    for cam in cameras:
        path = IMAGES_PATH_RAW / f"camera_{cam}.png"

        frame = cv2.imread(str(path))

        if frame is None:
            raise FileNotFoundError(f"Imagem não encontrada: {path}")

        images_by_cam[cam] = frame

    return images_by_cam


def extract_annotations(result):
    annotations = []

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

        annotation = {
            "id": i,
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": round(confidence, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        }

        annotations.append(annotation)

    return annotations


def save_annotations(cam_id, frame, annotations):
    ANNOTATIONS_PATH.mkdir(parents=True, exist_ok=True)

    height, width = frame.shape[:2]

    data = {
        "camera_id": cam_id,
        "image": f"camera_{cam_id}.png",
        "width": width,
        "height": height,
        "num_objects": len(annotations),
        "annotations": annotations
    }

    output_path = ANNOTATIONS_PATH / f"camera_{cam_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Camera {cam_id}: {len(annotations)} objetos detectados")
    print(f"JSON salvo em: {output_path}")


def main():
    cameras = [1, 2, 3, 4]

    images_by_cam = load_images(cameras)

    model = YOLO(os.environ.get("YOLO_MODEL_PATH", str(PROJECT_ROOT / "yolo26n.pt")))

    for cam in cameras:
        frame = images_by_cam[cam]

        result = model(frame, verbose=False)[0]

        annotations = extract_annotations(result)

        save_annotations(cam, frame, annotations)


if __name__ == "__main__":
    main()
