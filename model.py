from ultralytics import YOLO
from PIL import Image
import os
import glob


class ObjectDetector:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    # ---------------- IMAGE DETECTION ----------------

    def predict_image(self, image):

        image = image.convert("RGB")

        results = self.model.predict(
            source=image,
            conf=0.7,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "label": self.model.names[int(box.cls)],
                    "confidence": round(float(box.conf), 2),
                    "box": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ]
                })

        return detections

    # ---------------- VIDEO DETECTION ----------------

    def predict_video(self, video_path):

        results = self.model.predict(
            source=video_path,
            conf=0.7,
            save=True,
            project="runs",
            name="detect",
            exist_ok=True,
            verbose=False
        )

        save_dir = str(results[0].save_dir)

        print("Save Dir:", save_dir)

        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]

        video_files = []

        for ext in video_extensions:
            video_files.extend(
                glob.glob(os.path.join(save_dir, ext))
            )

        if not video_files:
            raise FileNotFoundError(
                f"No detected video found inside: {save_dir}"
            )

        latest_video = max(video_files, key=os.path.getctime)

        print("Detected Video:", latest_video)

        return latest_video


detector = ObjectDetector()