# Entry point: runs the full pipeline — detection → serialization → LLM summary → annotated video

from dotenv import load_dotenv
load_dotenv()

import argparse
from src.detector import VideoDetector
from src.tracker import CentroidTracker
from src.serializer import SceneSerializer
from src.llm_client import LLMClient
from src.visualizer import render_annotated_video
from src.utils import save_json, log


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--output", default="outputs/scene_output.json")
    p.add_argument("--min-frames", type=int, default=2)
    return p.parse_args()


def run(args):
    log("Step 1: Detection + Tracking")
    detector = VideoDetector(args.model, args.conf, args.fps)
    raw_tracks = detector.process_video(args.video, CentroidTracker())

    log("Step 2: Serialization")
    scene = SceneSerializer(args.min_frames).serialize(raw_tracks, detector.video_fps, detector.frame_size)
    save_json(scene, args.output)

    log("Step 3: LLM Summary")
    scene["llm_summary"] = LLMClient().generate_summary(scene)
    save_json(scene, args.output)
    print("\n--- SUMMARY ---\n" + scene["llm_summary"] + "\n")

    log("Step 4: Annotated Video")
    render_annotated_video(args.video, raw_tracks, args.output.replace(".json", "_annotated.mp4"))


if __name__ == "__main__":
    run(parse_args())