# Renders a copy of the video with color-coded bounding boxes, labels, and confidence scores.

import cv2
import os

# One distinct color per class (BGR)
CLASS_COLORS = {
    "car":        (235, 160,  50),
    "bus":        ( 60, 180,  75),
    "truck":      ( 67, 114, 196),
    "person":     ( 76,  76, 255),
    "bicycle":    (148,  30, 230),
    "motorcycle": ( 30, 220, 220),
}
DEFAULT_COLOR = (200, 200, 200)


def render_annotated_video(video_path, raw_tracks, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Build frame_idx → list of (cls, conf, x1, y1, x2, y2)
    frames_raw = raw_tracks.get("_frames_raw", {})

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = frames_raw.get(frame_idx, [])
        for cls, conf, x1, y1, x2, y2 in boxes:
            color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")