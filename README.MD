# Multi-Modal Scene Understanding Pipeline

An end-to-end pipeline that takes a video, detects and tracks objects using YOLOv8, serializes the spatial/temporal data, and generates a natural language scene summary using Gemini.

---

## Project Structure

```
multimodal_pipeline/
├── main.py
├── requirements.txt
├── .env
├── outputs/
│   ├── scene_output.json
│   └── scene_output_annotated.mp4
└── src/
    ├── detector.py       # Frame sampling + YOLO detection
    ├── tracker.py        # Centroid-based object tracker
    ├── serializer.py     # Raw tracks → structured JSON + anomaly detection
    ├── llm_client.py     # Gemini API call
    ├── prompt_builder.py # Builds LLM prompt from scene data
    ├── visualizer.py     # Renders annotated video with bounding boxes
    └── utils.py          # Logging, JSON save/load
```

---

## Setup

```bash

python -m venv venv
venv\Scripts\activate      

pip install -r requirements.txt
```

Add your Gemini API key to `.env` (get one free at https://aistudio.google.com/app/apikey):
```
GEMINI_API_KEY=your_key_here
```

---

## Run

```bash
python main.py --video path/to/video.mp4
```

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--fps` | `2.0` | Frames per second to sample |
| `--conf` | `0.4` | YOLO detection confidence threshold |
| `--model` | `yolov8n.pt` | YOLO weights (auto-downloaded) |
| `--output` | `outputs/scene_output.json` | Output JSON path |
| `--min-frames` | `2` | Min frames an object must appear (noise filter) |

---
