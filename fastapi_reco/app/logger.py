import os
import json
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "recommendation_log.jsonl")

def save_log(log: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    log["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")
