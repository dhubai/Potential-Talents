from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeedbackStore:
    path: Path

    def load(self) -> dict:
        if not self.path.exists():
            return {"events": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def append_event(self, event: dict) -> None:
        data = self.load()
        data["events"].append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

