# Samples frames from a video and runs YOLO detection on each, returning raw tracking data.

import cv2
from ultralytics import YOLO
from collections import defaultdict

RELEVANT_CLASSES = {"person", "car", "bus", "truck", "bicycle", "motorcycle"}


class VideoDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4, sample_fps=2.0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.sample_fps = sample_fps
        self.video_fps = None
        self.frame_size = None

    def process_video(self, video_path, tracker):
        cap = cv2.VideoCapture(video_path)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (w, h)
        interval = max(1, int(self.video_fps / self.sample_fps))

        raw_tracks = defaultdict(lambda: {"class": None, "frames": [], "raw_boxes": []})
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                ts = round(frame_idx / self.video_fps, 2)
                detections, raw = self._detect(frame)
                for obj_id, cls, bbox in tracker.update(detections):
                    raw_tracks[obj_id]["class"] = cls
                    raw_tracks[obj_id]["frames"].append((frame_idx, ts, bbox))
                    
                raw_tracks["_frames_raw"] = raw_tracks.get("_frames_raw", {})
                raw_tracks["_frames_raw"][frame_idx] = raw
            frame_idx += 1

        cap.release()
        return dict(raw_tracks)

    def _detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections, raw_boxes = [], []
        h, w = frame.shape[:2]
        for box in results.boxes:
            cls = self.model.names[int(box.cls)]
            if cls not in RELEVANT_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = round(float(box.conf), 2)
            bbox_norm = (round((x1+x2)/2/w, 3), round((y1+y2)/2/h, 3), round((x2-x1)/w, 3), round((y2-y1)/h, 3))
            detections.append((cls, bbox_norm))
            raw_boxes.append((cls, conf, int(x1), int(y1), int(x2), int(y2)))
        return detections, raw_boxes