# Converts raw tracking data into a structured scene JSON for LLM input.

ZONES = [(0.33, "left"), (0.66, "center"), (1.0, "right")]


def get_zone(cx):
    return next(name for thresh, name in ZONES if cx <= thresh)


def estimate_speed(frames):
    if len(frames) < 2:
        return None
    dists = []
    for i in range(1, len(frames)):
        _, t0, b0 = frames[i-1]; _, t1, b1 = frames[i]
        dt = t1 - t0
        if dt > 0:
            dists.append(((b1[0]-b0[0])**2 + (b1[1]-b0[1])**2)**0.5 / dt)
    return round(sum(dists) / len(dists), 4) if dists else None


class SceneSerializer:
    def __init__(self, min_frames=2, anomaly_threshold=0.08):
        self.min_frames = min_frames
        self.anomaly_threshold = anomaly_threshold

    def serialize(self, raw_tracks, video_fps, frame_size):
        objects, anomalies, counters = [], [], {}

        for obj_id, track in raw_tracks.items():
            if obj_id == "_frames_raw" or not track["frames"] or len(track["frames"]) < self.min_frames:
                continue

            cls, frames = track["class"], track["frames"]
            counters[cls] = counters.get(cls, 0) + 1
            name = f"{cls.capitalize()}_{counters[cls]}"

            fz, lz = get_zone(frames[0][2][0]), get_zone(frames[-1][2][0])
            movement = f"remained in the {fz} of the frame" if fz == lz else f"moved from {fz} to {lz}"
            speed = estimate_speed(frames)

            obj = {
                "id": name, "class": cls,
                "first_seen": frames[0][1], "last_seen": frames[-1][1],
                "duration_sec": round(frames[-1][1] - frames[0][1], 2),
                "frame_count": len(frames),
                "movement": movement,
                "speed_norm_per_sec": speed,
            }

            if speed and speed > self.anomaly_threshold:
                ts = lambda s: f"{int(s//3600):02}:{int(s%3600//60):02}:{int(s%60):02}:{int((s%1)*100):02}"
                anomalies.append({"object": name, "type": "high_speed",
                                  "description": f"{name} moved at unusually high speed from {ts(frames[0][1])} to {ts(frames[-1][1])}",
                                  "speed": speed})
                obj["anomaly"] = "high_speed"

            objects.append(obj)

        counts = {}
        for o in objects: counts[o["class"]] = counts.get(o["class"], 0) + 1

        return {
            "video_info": {"fps": video_fps, "width": frame_size[0], "height": frame_size[1]},
            "summary_counts": counts,
            "objects": objects,
            "anomalies": anomalies,
        }