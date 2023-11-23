import json
import re

from pathlib import Path

module_dir = Path(__file__).parent

project_root = module_dir.parent.parent

with open(project_root / "resources/modality-patterns.json") as f:
    modality_patterns = json.loads(f.read())


def split_defs(s: str) -> list[str]:
    defs = s.split("（")
    defs = [re.sub(r"[（）]", "", d) for d in defs]
    return defs


# TODO ものの as 文中接続表現
with open(project_root / "resources/connectives-patterns.json") as f:
    connectives_patterns = json.loads(f.read())
    connectives_classifications = {
        sub_pattern: pattern["kinou"]
        for pattern in connectives_patterns
        for sub_pattern in split_defs(pattern["conjunction"])
    }

with open(project_root / "resources/connectives-regexes.json") as f:
    connectives_regexes = json.loads(f.read())
    connectives_classifications = {
        sub_pattern: pattern["kinou"]
        for pattern in connectives_regexes
        for sub_pattern in split_defs(pattern["conjunction"])
    }

if __name__ == "__main__":
    print(modality_patterns)
    print(connectives_patterns)
    print(connectives_regexes)
    print(connectives_classifications)
