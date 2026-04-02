from __future__ import annotations

import json
from pathlib import Path


def sanitize_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cleaned = []

    for cell in nb.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = str(source)

        if not source_text.strip():
            continue

        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

        cleaned.append(cell)

    nb["cells"] = cleaned
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sanitize_notebook(Path("PotentialTalents.ipynb"))
