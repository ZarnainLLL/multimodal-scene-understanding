"""
Builds structured system + user prompts from scene data for the LLM.
This is the "bridge" that translates spatial/temporal tracking data
into natural language context the LLM can reason about.
"""

import json


SYSTEM_PROMPT = """You are a traffic and scene analysis AI. You will receive structured data \
from a computer vision pipeline that tracked objects in a video. The data includes:
- Object IDs (e.g., Person_1, Car_2)
- When each object first appeared and last appeared (timestamps in seconds)
- Which horizontal zone they were in (left, center, right of frame)
- Their movement across zones
- Any detected anomalies 

Your task:
1. Write a concise 2-3 sentence summary answering: "What are the main events happening in this video clip?"
2. If any anomalies are present, include ONE sentence highlighting them.
3. Be specific: mention object counts, directions, and notable interactions.
4. Do NOT mention raw numbers like bounding box coords or normalized values."""


def build_prompt(scene_data: dict) -> tuple[str, str]:
    counts = scene_data.get("summary_counts", {})
    objects = scene_data.get("objects", [])
    anomalies = scene_data.get("anomalies", [])

    # Build object descriptions
    obj_lines = []
    for obj in objects:
        line = (
            f"- {obj['id']}: appeared at {obj['first_seen']}s, "
            f"last seen at {obj['last_seen']}s, "
            f"{obj['movement']}"
        )
        if obj.get("anomaly"):
            line += " [ANOMALY: high speed]"
        obj_lines.append(line)

    anomaly_text = ""
    if anomalies:
        anomaly_lines = [f"  * {a['description']}" for a in anomalies]
        anomaly_text = "\nANOMALIES DETECTED:\n" + "\n".join(anomaly_lines)

    user_prompt = f"""Scene Object Summary:
Total objects tracked: {len(objects)}
Class breakdown: {json.dumps(counts)}

Object-level tracking data:
{chr(10).join(obj_lines)}
{anomaly_text}
"""

    return SYSTEM_PROMPT, user_prompt