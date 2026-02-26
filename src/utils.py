# Shared utilities: logging, JSON save/load.

import json, logging, os

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

def log(msg): logging.info(msg)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w"), indent=2)

def load_json(path):
    return json.load(open(path))